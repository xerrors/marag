#!/usr/bin/env python3
"""交互式浏览 predictions.jsonl 文件

Usage: python browse_predictions.py <predictions.jsonl>

操作:
  ↑/↓ 或 j/k   上下切换
  D            删除当前条目
  Q            退出
"""

import json
import sys
import os

# ANSI 颜色码
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def load_predictions(filepath: str) -> list[dict]:
    """加载 predictions.jsonl 文件"""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_predictions(filepath: str, data: list[dict]):
    """保存 predictions.jsonl 文件"""
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def clear_screen():
    """清屏"""
    os.system("cls" if os.name == "nt" else "clear")


def wrap(text: str, width: int = 76) -> list[str]:
    """简单自动换行"""
    lines = []
    for paragraph in text.split('\n'):
        while len(paragraph) > width:
            lines.append(paragraph[:width])
            paragraph = paragraph[width:]
        if paragraph:
            lines.append(paragraph)
    return lines


def highlight_text(text: str, gold: str) -> str:
    """如果 gold 出现在 text 中，高亮显示（绿色）"""
    if gold and text and gold in text:
        return text.replace(gold, f"{Colors.GREEN}{gold}{Colors.RESET}")
    return text


def display_item(item: dict, index: int, total: int, deleted: bool = False):
    """显示单个条目"""
    clear_screen()

    # 顶部状态栏
    deleted_mark = f" {Colors.RED}[已删除]{Colors.RESET}" if deleted else ""
    progress = f"{Colors.CYAN}{index + 1}{Colors.RESET} / {total}"
    print(f"{Colors.BOLD}{'─' * 78}{Colors.RESET}")
    print(f"{Colors.BOLD}  {progress}  {Colors.RESET}{deleted_mark}")
    print(f"{Colors.BOLD}{'─' * 78}{Colors.RESET}")

    # Question
    print(f"\n{Colors.BOLD}{Colors.CYAN}▸ Question:{Colors.RESET}")
    for line in wrap(item.get('question', 'N/A'), 76):
        print(f"  {line}")

    # Gold Answer
    gold = item.get('gold_answer') or ''
    print(f"\n{Colors.BOLD}{Colors.YELLOW}▸ Gold Answer:{Colors.RESET}")
    for line in wrap(gold, 76):
        print(f"  {Colors.YELLOW}{line}{Colors.RESET}")

    # Pred Answer
    pred = item.get('pred_answer') or 'N/A'
    highlighted_pred = highlight_text(pred, gold)
    print(f"\n{Colors.BOLD}{Colors.BLUE}▸ Pred Answer:{Colors.RESET}")
    for line in wrap(highlighted_pred, 76):
        print(f"  {line}")

    # 指标区域
    print(f"\n{Colors.BOLD}{'─' * 78}{Colors.RESET}")

    loops = item.get('loops', 'N/A')
    chunks = item.get('chunks_read_count', 'N/A')
    llm_acc = item.get('llm_accuracy', 'N/A')
    contain = item.get('contain_accuracy', 'N/A')

    # 根据正确性着色
    llm_color = Colors.GREEN if llm_acc == 1.0 else Colors.RED if llm_acc == 0.0 else Colors.YELLOW
    contain_color = Colors.GREEN if contain == 1 else Colors.RED if contain == 0 else Colors.YELLOW

    print(f"  {Colors.DIM}Loops:{Colors.RESET} {loops}  |  {Colors.DIM}Chunks:{Colors.RESET} {chunks}  |  "
          f"{Colors.DIM}LLM ACC:{Colors.RESET} {llm_color}{llm_acc}{Colors.RESET}  |  "
          f"{Colors.DIM}Contain:{Colors.RESET} {contain_color}{contain}{Colors.RESET}")

    print(f"{Colors.BOLD}{'─' * 78}{Colors.RESET}")
    print(f"{Colors.DIM}  ↑/↓ j/k 切换  |  D 删除  |  Q 退出{Colors.RESET}")


def get_key():
    """获取单个按键，支持方向键"""
    if os.name == "nt":
        import msvcrt
        ch = msvcrt.getch()
        if ch == b'\xe0':  # 方向键前缀
            ch = msvcrt.getch()
            if ch == b'H':
                return '\x1b[A'  # 上
            elif ch == b'P':
                return '\x1b[B'  # 下
        return ch.decode('utf-8', errors='ignore')
    else:
        import tty
        import termios

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            if ch == '\x1b':  # ESC
                # 可能是一串转义序列
                ch += sys.stdin.read(2)
            return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <predictions.jsonl>")
        sys.exit(1)

    filepath = sys.argv[1]
    if not os.path.exists(filepath):
        print(f"文件不存在: {filepath}")
        sys.exit(1)

    data = load_predictions(filepath)
    if not data:
        print("文件为空")
        sys.exit(1)

    current = 0
    deleted = set()
    redraw = True

    # 初始化
    print(f"\n{Colors.GREEN}加载了 {len(data)} 条记录{Colors.RESET}")
    print(f"{Colors.DIM}按任意键开始浏览...{Colors.RESET}")
    get_key()

    while True:
        if redraw:
            display_item(data[current], current, len(data), current in deleted)

        try:
            key = get_key()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if key == 'q' or key == 'Q':
            break
        elif key == 'd' or key == 'D':
            if current not in deleted:
                deleted.add(current)
                print(f"\n{Colors.RED}✗ 已标记删除: {current + 1}{Colors.RESET}")
            else:
                deleted.discard(current)
                print(f"\n{Colors.GREEN}✓ 已取消删除: {current + 1}{Colors.RESET}")
            redraw = True
        elif key in ('\x1b[A', 'k', 'K'):  # 上 或 k
            if current > 0:
                current -= 1
            redraw = True
        elif key in ('\x1b[B', 'j', 'J'):  # 下 或 j
            if current < len(data) - 1:
                current += 1
            redraw = True
        else:
            redraw = True

    # 保存删除操作
    if deleted:
        print(f"\n{Colors.YELLOW}正在保存... (删除了 {len(deleted)} 条){Colors.RESET}")
        # 反向排序，依次删除
        for idx in sorted(deleted, reverse=True):
            del data[idx]
        save_predictions(filepath, data)
        print(f"{Colors.GREEN}✓ 保存完成{Colors.RESET}")
    else:
        print(f"\n{Colors.DIM}没有修改，退出{Colors.RESET}")


if __name__ == "__main__":
    main()
