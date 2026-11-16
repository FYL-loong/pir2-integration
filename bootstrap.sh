#!/usr/bin/env bash
# 从零重建 πR² on π0.5 的工作环境（容器销毁后用它恢复）。
#
# 用法：
#   bash bootstrap.sh                # 默认装到 /workspace
#   WORK=/data/xxx bash bootstrap.sh # 指定目录
#
# 需要：能访问 github.com 和 pypi（容器已验证可访问）。

set -euo pipefail

WORK="${WORK:-/workspace}"
REPO="${REPO:-https://github.com/FYL-loong/pir2-integration.git}"
OPENPI="${OPENPI:-https://github.com/Physical-Intelligence/openpi.git}"
INDEX="${INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"

echo "==> 工作目录: $WORK"
mkdir -p "$WORK"
cd "$WORK"

# 1) uv
if ! command -v uv >/dev/null 2>&1; then
  echo "==> 安装 uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # shellcheck disable=SC1091
  source "$HOME/.local/bin/env" 2>/dev/null || source "$HOME/.bashrc" || true
fi
export PATH="$HOME/.local/bin:$PATH"
export UV_DEFAULT_INDEX="$INDEX"

# 2) openpi（不拉 submodule：aloha/libero 我们不需要，能省 350 MB 和几分钟）
if [ ! -d "$WORK/openpi/.git" ]; then
  echo "==> clone openpi"
  git clone "$OPENPI" "$WORK/openpi"
fi

# 3) 我们的集成包
if [ ! -d "$WORK/pir2_integration/.git" ]; then
  echo "==> clone pir2-integration"
  git clone "$REPO" "$WORK/pir2_integration"
else
  echo "==> 更新 pir2-integration"
  git -C "$WORK/pir2_integration" pull --ff-only || true
fi

# 4) 依赖
echo "==> uv sync（约 3 GB / 9 分钟）"
cd "$WORK/openpi"
uv sync

# 5) 打补丁（幂等，可重复执行）
echo "==> 打补丁"
python "$WORK/pir2_integration/apply_patch.py" --openpi-dir "$WORK/openpi" --pi05-state

# 6) 自检
echo "==> 自检"
uv run python "$WORK/pir2_integration/preflight.py"

echo
echo "完成。下一步："
echo "  cd $WORK/openpi"
echo "  uv run python $WORK/pir2_integration/preflight.py --gates   # 会加载 3B 模型，3~10 分钟"
