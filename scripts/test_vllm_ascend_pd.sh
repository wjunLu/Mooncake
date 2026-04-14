#!/bin/bash
# Test Mooncake Ascend wheel with vLLM PD (Prefill-Decode) disaggregation.
#
# Prerequisites:
#   - Ascend NPU hardware (Atlas 800T A2, 910B, etc.) with /dev/davinci* devices
#   - CANN toolkit >= 8.5.1 installed and sourced
#   - vllm-ascend installed (pip install vllm-ascend)
#   - Mooncake Ascend wheel installed (pip install mooncake-transfer-engine-ascend)
#   - A local model or HuggingFace model path
#
# Usage:
#   bash test_vllm_ascend_pd.sh [MODEL_PATH] [PREFILL_DEVICE] [DECODE_DEVICE]
#
# Examples:
#   # Use Qwen2.5-7B on NPU 0 (prefill) and NPU 1 (decode)
#   bash test_vllm_ascend_pd.sh /model/Qwen2.5-7B-Instruct 0 1
#
#   # Auto-detect first two available NPUs
#   bash test_vllm_ascend_pd.sh /model/Qwen2.5-7B-Instruct

set -e

# ── Configuration ────────────────────────────────────────────────────────────
MODEL_PATH="${1:-Qwen/Qwen2.5-7B-Instruct}"
PREFILL_DEVICE="${2:-0}"
DECODE_DEVICE="${3:-1}"
NODE_IP="${NODE_IP:-127.0.0.1}"
PREFILL_PORT="${PREFILL_PORT:-13700}"
DECODE_PORT="${DECODE_PORT:-13701}"
PROXY_PORT="${PROXY_PORT:-8080}"
PREFILL_KV_PORT="${PREFILL_KV_PORT:-30000}"
DECODE_KV_PORT="${DECODE_KV_PORT:-30100}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
LOG_DIR="/tmp/mooncake-pd-test-$$"
VLLM_ASCEND_VERSION="${VLLM_ASCEND_VERSION:-main}"

# ── Helpers ──────────────────────────────────────────────────────────────────
info()  { echo "[INFO]  $*"; }
ok()    { echo "[OK]    $*"; }
fail()  { echo "[FAIL]  $*" >&2; exit 1; }

cleanup() {
    info "Cleaning up processes..."
    [ -n "$PREFILL_PID" ] && kill "$PREFILL_PID" 2>/dev/null || true
    [ -n "$DECODE_PID"  ] && kill "$DECODE_PID"  2>/dev/null || true
    [ -n "$PROXY_PID"   ] && kill "$PROXY_PID"   2>/dev/null || true
    wait 2>/dev/null || true
    info "Logs saved to $LOG_DIR"
}
trap cleanup EXIT

mkdir -p "$LOG_DIR"

# ── Pre-flight checks ─────────────────────────────────────────────────────────
info "=== Mooncake Ascend PD Disaggregation Test ==="
info "Model:          $MODEL_PATH"
info "Prefill device: $PREFILL_DEVICE  (port $PREFILL_PORT, KV port $PREFILL_KV_PORT)"
info "Decode device:  $DECODE_DEVICE   (port $DECODE_PORT,  KV port $DECODE_KV_PORT)"
info "Proxy:          $NODE_IP:$PROXY_PORT"
echo

# Check mooncake wheel
python3 -c "import mooncake" 2>/dev/null \
    || fail "mooncake not installed. Run: pip install mooncake-transfer-engine-ascend"
MOONCAKE_VER=$(python3 -c "import importlib.metadata; print(importlib.metadata.version('mooncake-transfer-engine-ascend'))" 2>/dev/null || echo "unknown")
ok "mooncake wheel: $MOONCAKE_VER"

# Check vllm
python3 -c "import vllm" 2>/dev/null \
    || fail "vllm not installed. Run: pip install vllm-ascend"
VLLM_VER=$(python3 -c "import importlib.metadata; print(importlib.metadata.version('vllm'))" 2>/dev/null || echo "unknown")
ok "vllm: $VLLM_VER"

# Check Ascend devices
[ -e "/dev/davinci${PREFILL_DEVICE}" ] \
    || fail "Ascend device /dev/davinci${PREFILL_DEVICE} not found"
[ -e "/dev/davinci${DECODE_DEVICE}" ] \
    || fail "Ascend device /dev/davinci${DECODE_DEVICE} not found"
ok "Ascend NPU devices found"

# Check proxy script
PROXY_SCRIPT=$(python3 -c "
import sys, vllm, pathlib
vllm_root = pathlib.Path(vllm.__file__).parent
proxy = vllm_root / 'tests/v1/kv_connector/nixl_integration/toy_proxy_server.py'
print(proxy if proxy.exists() else '')
" 2>/dev/null)
if [ -z "$PROXY_SCRIPT" ]; then
    # Fall back to vllm-ascend's proxy
    PROXY_SCRIPT=$(python3 -c "
import subprocess, pathlib
r = subprocess.run(['pip', 'show', '-f', 'vllm-ascend'], capture_output=True, text=True)
for line in r.stdout.splitlines():
    if 'load_balance_proxy_server_example.py' in line:
        loc = subprocess.run(['pip', 'show', 'vllm-ascend'], capture_output=True, text=True)
        for l in loc.stdout.splitlines():
            if l.startswith('Location:'):
                base = l.split()[-1]
                import pathlib
                print(pathlib.Path(base) / 'vllm_ascend' / line.strip())
                break
" 2>/dev/null)
fi
[ -n "$PROXY_SCRIPT" ] && ok "Proxy script: $PROXY_SCRIPT" \
    || fail "Could not locate proxy server script. Install vllm-ascend or vllm."

# ── Start Prefiller ───────────────────────────────────────────────────────────
info "Starting prefill server on device $PREFILL_DEVICE port $PREFILL_PORT ..."
export ASCEND_RT_VISIBLE_DEVICES="$PREFILL_DEVICE"
export HCCL_IF_IP="$NODE_IP"
export OMP_NUM_THREADS=4

vllm serve "$MODEL_PATH" \
    --host "$NODE_IP" \
    --port "$PREFILL_PORT" \
    --tensor-parallel-size 1 \
    --enforce-eager \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens "$MAX_MODEL_LEN" \
    --gpu-memory-utilization 0.85 \
    --kv-transfer-config \
    "{\"kv_connector\":\"MooncakeConnectorV1\",
      \"kv_role\":\"kv_producer\",
      \"kv_port\":\"${PREFILL_KV_PORT}\",
      \"engine_id\":\"0\",
      \"kv_buffer_device\":\"npu\",
      \"kv_connector_extra_config\":{
          \"prefill\":{\"dp_size\":1,\"tp_size\":1},
          \"decode\":{\"dp_size\":1,\"tp_size\":1}}}" \
    > "$LOG_DIR/prefill.log" 2>&1 &
PREFILL_PID=$!
unset ASCEND_RT_VISIBLE_DEVICES

# ── Start Decoder ─────────────────────────────────────────────────────────────
info "Starting decode server on device $DECODE_DEVICE port $DECODE_PORT ..."
export ASCEND_RT_VISIBLE_DEVICES="$DECODE_DEVICE"

vllm serve "$MODEL_PATH" \
    --host "$NODE_IP" \
    --port "$DECODE_PORT" \
    --tensor-parallel-size 1 \
    --enforce-eager \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens "$MAX_MODEL_LEN" \
    --gpu-memory-utilization 0.85 \
    --kv-transfer-config \
    "{\"kv_connector\":\"MooncakeConnectorV1\",
      \"kv_role\":\"kv_consumer\",
      \"kv_port\":\"${DECODE_KV_PORT}\",
      \"engine_id\":\"1\",
      \"kv_buffer_device\":\"npu\",
      \"kv_connector_extra_config\":{
          \"prefill\":{\"dp_size\":1,\"tp_size\":1},
          \"decode\":{\"dp_size\":1,\"tp_size\":1}}}" \
    > "$LOG_DIR/decode.log" 2>&1 &
DECODE_PID=$!
unset ASCEND_RT_VISIBLE_DEVICES

# ── Wait for servers to be ready ──────────────────────────────────────────────
wait_server() {
    local name="$1" url="$2"
    info "Waiting for $name to be ready ..."
    for i in $(seq 1 120); do
        if curl -sf "$url/health" >/dev/null 2>&1; then
            ok "$name is ready"
            return 0
        fi
        # Also check if the process died
        if ! kill -0 "$3" 2>/dev/null; then
            fail "$name process died. Check $LOG_DIR/${name}.log"
        fi
        sleep 2
    done
    fail "$name did not become ready within 240s. Check $LOG_DIR/${name}.log"
}

wait_server "prefill" "http://${NODE_IP}:${PREFILL_PORT}" "$PREFILL_PID"
wait_server "decode"  "http://${NODE_IP}:${DECODE_PORT}"  "$DECODE_PID"

# ── Start Proxy ───────────────────────────────────────────────────────────────
info "Starting proxy server on port $PROXY_PORT ..."
python3 "$PROXY_SCRIPT" \
    --host "$NODE_IP" \
    --port "$PROXY_PORT" \
    --prefiller-hosts "$NODE_IP" \
    --prefiller-ports "$PREFILL_PORT" \
    --decoder-hosts "$NODE_IP" \
    --decoder-ports "$DECODE_PORT" \
    > "$LOG_DIR/proxy.log" 2>&1 &
PROXY_PID=$!

sleep 3
kill -0 "$PROXY_PID" 2>/dev/null || fail "Proxy process died. Check $LOG_DIR/proxy.log"
ok "Proxy server started"

# ── Send test inference request ───────────────────────────────────────────────
info "Sending inference requests via proxy ..."

MODEL_NAME=$(python3 -c "print('$MODEL_PATH'.split('/')[-1])")

# Short prompt — basic connectivity test
RESP=$(curl -sf "http://${NODE_IP}:${PROXY_PORT}/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"${MODEL_PATH}\",
        \"prompt\": \"Hello, world!\",
        \"max_tokens\": 16
    }" 2>&1) || fail "Short-prompt request failed"
echo "$RESP" | python3 -c "
import sys, json
d = json.load(sys.stdin)
text = d['choices'][0]['text']
print(f'  Response (short): {text!r}')
assert len(text) > 0, 'Empty response'
" || fail "Response parsing failed"
ok "Short-prompt inference passed"

# Long prompt — exercises KV cache transfer (the key Mooncake feature)
LONG_PROMPT="Given the accelerating impacts of climate change—including rising sea levels, \
increasing frequency of extreme weather events, loss of biodiversity, and adverse effects \
on agriculture and human health—there is an urgent need for a robust, globally coordinated \
response. How can global agreements like the Paris Accord be strengthened to enforce \
emission reduction targets while promoting fair technology transfer to vulnerable regions?"

RESP=$(curl -sf "http://${NODE_IP}:${PROXY_PORT}/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"${MODEL_PATH}\",
        \"prompt\": \"${LONG_PROMPT}\",
        \"max_tokens\": 128
    }" 2>&1) || fail "Long-prompt request failed"
echo "$RESP" | python3 -c "
import sys, json
d = json.load(sys.stdin)
text = d['choices'][0]['text']
usage = d.get('usage', {})
print(f'  Response (long): {text[:80]!r}...')
print(f'  Tokens: prompt={usage.get(\"prompt_tokens\")}, completion={usage.get(\"completion_tokens\")}')
assert len(text) > 0, 'Empty response'
" || fail "Long-prompt response parsing failed"
ok "Long-prompt inference passed (KV transfer exercised)"

# ── Summary ───────────────────────────────────────────────────────────────────
echo
echo "============================================"
echo "  All inference tests PASSED"
echo "  Mooncake version: $MOONCAKE_VER"
echo "  vLLM version:     $VLLM_VER"
echo "  Model:            $MODEL_PATH"
echo "  Architecture:     1P1D (single-node)"
echo "============================================"
