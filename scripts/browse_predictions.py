#!/usr/bin/env python3
"""交互式浏览 predictions.jsonl 文件

Usage: python browse_predictions.py <predictions.jsonl>

操作:
  ↑/↓ 或 j/k   上下切换
  D            删除当前条目
  V            切换 Detail / 简洁模式
  Q            退出
"""

import json
import re
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


def get_terminal_width(min_width: int = 80, max_width: int = 140) -> int:
    """获取终端可用宽度，留 4 字符边距"""
    try:
        cols = os.get_terminal_size().columns
    except OSError:
        cols = 100
    return max(min_width, min(cols, max_width)) - 4


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


def wrap(text: str, width: int) -> list[str]:
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
    """如果 gold 出现在 text 中（大小写不敏感），高亮显示（绿色）"""
    if gold and text:
        return re.sub(
            re.escape(gold),
            lambda m: f"{Colors.GREEN}{m.group(0)}{Colors.RESET}",
            text,
            flags=re.IGNORECASE,
        )
    return text


def format_args(args: dict, max_len: int = 80) -> str:
    """紧凑格式化参数"""
    if not args:
        return ""
    try:
        s = json.dumps(args, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        s = str(args)
    if len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return s


def truncate_result(text: str, max_chars: int = 320, max_lines: int = 6) -> str:
    """截断工具结果，控制显示长度"""
    if not text:
        return ""
    lines = text.splitlines()
    if len(lines) > max_lines:
        text = "\n".join(lines[:max_lines]) + "\n..."
    if len(text) > max_chars:
        text = text[: max_chars - 3] + "..."
    return text


def format_trajectory(trajectory: list[dict], width: int) -> list[str]:
    """将 trajectory 格式化为可打印的行列表"""
    lines = []
    if not trajectory:
        lines.append(f"{Colors.DIM}  (无轨迹数据){Colors.RESET}")
        return lines

    # 按 loop 分组（保持顺序）
    grouped = []
    current_loop = None
    group = []
    for entry in trajectory:
        loop = entry.get("loop", 0)
        if loop != current_loop:
            if group:
                grouped.append((current_loop, group))
            current_loop = loop
            group = []
        group.append(entry)
    if group:
        grouped.append((current_loop, group))

    for loop, entries in grouped:
        sep_width = max(4, width - 10 - len(str(loop)))
        lines.append(
            f"{Colors.DIM}{'──'} Loop {loop} {'─' * sep_width}{Colors.RESET}"
        )
        for entry in entries:
            entry_type = entry.get("type", "tool")
            if entry_type == "assistant":
                content = entry.get("content", "")
                if content:
                    prefix = f"{Colors.CYAN}  > {Colors.BOLD}思考:{Colors.RESET} {Colors.CYAN}"
                    wrapped = wrap(content, width - 10)
                    for i, line in enumerate(wrapped):
                        if i == 0:
                            lines.append(f"{prefix}{line}{Colors.RESET}")
                        else:
                            lines.append(f"      {Colors.CYAN}{line}{Colors.RESET}")
            else:
                tool_name = entry.get("tool_name", "unknown")
                args = entry.get("arguments", {})
                result = entry.get("tool_result", "")
                tokens = entry.get("retrieved_tokens", 0)
                chunks = entry.get("chunks_found", entry.get("new_chunks_count", "N/A"))
                error = entry.get("error")

                lines.append(
                    f"{Colors.YELLOW}  > {Colors.BOLD}{tool_name}{Colors.RESET}"
                )
                if args:
                    arg_str = format_args(args, max_len=max(width - 20, 40))
                    lines.append(f"{Colors.BLUE}    Args : {arg_str}{Colors.RESET}")
                if error:
                    lines.append(f"{Colors.RED}    Error: {error}{Colors.RESET}")

                meta_parts = []
                if isinstance(tokens, (int, float)) and tokens > 0:
                    meta_parts.append(f"{tokens} tokens")
                if chunks != "N/A":
                    meta_parts.append(f"{chunks} chunks")
                meta = f"{Colors.DIM}[{', '.join(meta_parts)}]{Colors.RESET}" if meta_parts else ""

                # detail 模式放宽截断
                max_chars = max(width * 4, 300)
                max_lines = max(width // 25, 5)
                truncated = truncate_result(result, max_chars=max_chars, max_lines=max_lines)
                if truncated:
                    if meta:
                        lines.append(f"    {meta}")
                    for line in wrap(truncated, width - 6):
                        lines.append(f"      {line}")
                else:
                    if meta:
                        lines.append(f"    {meta} (无结果内容)")
                    else:
                        lines.append(f"    {Colors.DIM}(无结果内容){Colors.RESET}")

    return lines


def display_item(item: dict, index: int, total: int, deleted: bool = False, width: int = 100):
    """简洁模式：显示单个条目"""
    clear_screen()

    # 顶部状态栏
    deleted_mark = f" {Colors.RED}[已删除]{Colors.RESET}" if deleted else ""
    progress = f"{Colors.CYAN}{index + 1}{Colors.RESET} / {total}"
    print(f"{Colors.BOLD}{'─' * width}{Colors.RESET}")
    print(f"{Colors.BOLD}  {progress}  {Colors.RESET}{deleted_mark}")
    print(f"{Colors.BOLD}{'─' * width}{Colors.RESET}")

    # Question
    print(f"\n{Colors.BOLD}{Colors.CYAN}▸ Question:{Colors.RESET}")
    for line in wrap(item.get('question', 'N/A'), width - 2):
        print(f"  {line}")

    # Gold Answer
    gold = item.get('gold_answer') or ''
    print(f"\n{Colors.BOLD}{Colors.YELLOW}▸ Gold Answer:{Colors.RESET}")
    for line in wrap(gold, width - 2):
        print(f"  {Colors.YELLOW}{line}{Colors.RESET}")

    # Pred Answer
    pred = item.get('pred_answer') or 'N/A'
    highlighted_pred = highlight_text(pred, gold)
    print(f"\n{Colors.BOLD}{Colors.BLUE}▸ Pred Answer:{Colors.RESET}")
    for line in wrap(highlighted_pred, width - 2):
        print(f"  {line}")

    # 指标区域
    print(f"\n{Colors.BOLD}{'─' * width}{Colors.RESET}")

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

    print(f"{Colors.BOLD}{'─' * width}{Colors.RESET}")
    print(f"{Colors.DIM}  ↑/↓ j/k 切换  |  D 删除  |  V 详情  |  Q 退出{Colors.RESET}")


def display_item_detail(item: dict, index: int, total: int, deleted: bool = False, width: int = 100):
    """Detail 模式：显示完整工具调用链"""
    clear_screen()

    deleted_mark = f" {Colors.RED}[已删除]{Colors.RESET}" if deleted else ""
    progress = f"{Colors.CYAN}{index + 1}{Colors.RESET} / {total}"
    print(f"{Colors.BOLD}{'─' * width}{Colors.RESET}")
    print(f"{Colors.BOLD}  {progress}  {Colors.RESET}{Colors.YELLOW}[DETAIL]{Colors.RESET}{deleted_mark}")
    print(f"{Colors.BOLD}{'─' * width}{Colors.RESET}")

    # Question
    print(f"\n{Colors.BOLD}{Colors.CYAN}Question:{Colors.RESET}")
    q_lines = wrap(item.get('question', 'N/A'), width - 2)
    for line in q_lines[:2]:
        print(f"  {line}")
    if len(q_lines) > 2:
        print(f"  {Colors.DIM}...{Colors.RESET}")

    # Trajectory
    print(f"\n{Colors.BOLD}{Colors.YELLOW}Trajectory:{Colors.RESET}")
    for line in format_trajectory(item.get('trajectory', []), width):
        print(line)

    # Gold / Pred（精简）
    gold = item.get('gold_answer') or ''
    pred = item.get('pred_answer') or 'N/A'
    gold_display = gold[:width - 10] + ('...' if len(gold) > width - 10 else '')
    pred_display = pred[:width * 2] + ('...' if len(pred) > width * 2 else '')
    print(f"\n{Colors.BOLD}{Colors.YELLOW}Gold:{Colors.RESET} {gold_display}")
    print(f"{Colors.BOLD}{Colors.BLUE}Pred:{Colors.RESET} {pred_display}")

    # 指标区域（增加 cost）
    print(f"\n{Colors.BOLD}{'─' * width}{Colors.RESET}")
    loops = item.get('loops', 'N/A')
    chunks = item.get('chunks_read_count', 'N/A')
    cost = item.get('total_cost', 'N/A')
    llm_acc = item.get('llm_accuracy', 'N/A')
    contain = item.get('contain_accuracy', 'N/A')

    llm_color = Colors.GREEN if llm_acc == 1.0 else Colors.RED if llm_acc == 0.0 else Colors.YELLOW
    contain_color = Colors.GREEN if contain == 1 else Colors.RED if contain == 0 else Colors.YELLOW
    cost_str = f"${cost:.6f}" if isinstance(cost, (int, float)) else str(cost)

    print(f"  {Colors.DIM}Loops:{Colors.RESET} {loops}  |  {Colors.DIM}Chunks:{Colors.RESET} {chunks}  |  "
          f"{Colors.DIM}Cost:{Colors.RESET} {cost_str}  |  "
          f"{Colors.DIM}LLM:{Colors.RESET} {llm_color}{llm_acc}{Colors.RESET}  |  "
          f"{Colors.DIM}Contain:{Colors.RESET} {contain_color}{contain}{Colors.RESET}")

    print(f"{Colors.BOLD}{'─' * width}{Colors.RESET}")
    print(f"{Colors.DIM}  ↑/↓ j/k 切换  |  D 删除  |  V 简洁  |  Q 退出{Colors.RESET}")


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
    detail_mode = False
    width = get_terminal_width()

    # 初始化
    print(f"\n{Colors.GREEN}加载了 {len(data)} 条记录{Colors.RESET}")
    print(f"{Colors.DIM}按任意键开始浏览...{Colors.RESET}")
    get_key()

    while True:
        if redraw:
            if detail_mode:
                display_item_detail(data[current], current, len(data), current in deleted, width)
            else:
                display_item(data[current], current, len(data), current in deleted, width)

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
        elif key == 'v' or key == 'V':
            detail_mode = not detail_mode
            mode_name = "Detail" if detail_mode else "简洁"
            print(f"\n{Colors.YELLOW}⇄ 切换模式: {mode_name}{Colors.RESET}")
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
