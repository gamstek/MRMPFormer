#!/usr/bin/env python3
"""
QuanFormer 环境依赖检测 — GUI 弹窗报告
========================================
用法: python check_gui.py

运行 check_env.py 后以图形弹窗展示检测结果，支持：
  - 可视化通过/失败/警告状态
  - Conda 环境管理（查找/创建 quanformer 环境）
  - 一键修复按钮
  - 支持用户指定已有 conda 环境名

依赖: Python 标准库 (tkinter)，无需额外安装。
"""

import json
import os
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# 路径常量
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CHECK_SCRIPT = Path(__file__).resolve().parent / "check_env.py"
FIX_SCRIPT = Path(__file__).resolve().parent.parent / "fix-dependencies" / "fix_env.py"
DEFAULT_ENV_NAME = "quanformer"


# ---------------------------------------------------------------------------
# 后端逻辑
# ---------------------------------------------------------------------------

def run_check(env_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """运行 check_env.py 并返回解析后的 JSON.

    使用 --outfile 写入临时文件，彻底避免 Windows 下 stdout 管道 GBK/UTF-8 编码错配。
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as tmp:
        tmp_path = tmp.name

    try:
        cmd = [sys.executable, str(CHECK_SCRIPT), "--json", "--outfile", tmp_path]
        if env_name:
            cmd.extend(["--target-env", env_name])

        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
            encoding="utf-8", errors="replace",
        )

        # 从临时文件读取 JSON（UTF-8 编码，不受管道编码影响）
        result_text = Path(tmp_path).read_text(encoding="utf-8")
        if result_text.strip():
            return json.loads(result_text)

        # 回退：尝试从 stdout 解析
        out = proc.stdout or ""
        json_start = out.find("{")
        if json_start >= 0:
            return json.loads(out[json_start:])
        return None
    except Exception:
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def list_conda_envs() -> List[str]:
    """列出所有 conda 环境名."""
    try:
        proc = subprocess.run(
            ["conda", "env", "list"], capture_output=True, text=True, timeout=30,
            encoding="utf-8", errors="replace",
        )
        envs = []
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts and not parts[0].startswith("/") and not parts[0].startswith("\\"):
                name = parts[0]
                if name not in ("base",) and name not in envs:
                    envs.append(name)
        return envs
    except Exception:
        return []


def create_conda_env(name: str = DEFAULT_ENV_NAME) -> bool:
    """创建 conda 环境."""
    try:
        proc = subprocess.run(
            f'conda create -n {name} python=3.11 -y',
            shell=True, capture_output=True, text=True, timeout=300,
            encoding="utf-8", errors="replace",
        )
        return proc.returncode == 0
    except Exception:
        return False


def run_fix(env_name: str, dry_run: bool = False) -> str:
    """运行修复脚本并返回输出."""
    cmd = [sys.executable, str(FIX_SCRIPT), "fix", env_name, "--yes"]
    if dry_run:
        cmd.append("--dry-run")
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
            encoding="utf-8", errors="replace",
        )
        return proc.stdout or proc.stderr or "(无输出)"
    except Exception as e:
        return f"执行失败: {e}"


# ---------------------------------------------------------------------------
# GUI 组件
# ---------------------------------------------------------------------------

COLORS = {
    "bg": "#1a1a2e",
    "bg2": "#16213e",
    "fg": "#e0e0e0",
    "fg_dim": "#a0a0b0",
    "accent": "#0f9b8e",
    "pass": "#2ecc71",
    "fail": "#e74c3c",
    "warn": "#f39c12",
    "btn_bg": "#0f3460",
    "btn_fg": "#e0e0e0",
    "btn_hover": "#1a5276",
    "row_even": "#1e2d4a",
    "row_odd": "#1a2740",
}

FONT_TITLE = ("Microsoft YaHei UI", 13, "bold")
FONT_HEADING = ("Microsoft YaHei UI", 11, "bold")
FONT_BODY = ("Microsoft YaHei UI", 10)
FONT_MONO = ("Consolas", 9)
FONT_BTN = ("Microsoft YaHei UI", 10, "bold")


class CheckGUI:
    """环境检测报告弹窗."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("QuanFormer — 环境依赖检测报告")
        self.root.geometry("860x680")
        self.root.minsize(700, 500)
        self.root.configure(bg=COLORS["bg"])

        # 状态变量
        self.check_result: Optional[Dict[str, Any]] = None
        self.target_env: str = ""
        self.conda_envs: List[str] = []
        self.fix_output: str = ""

        # 居中窗口
        self.root.update_idletasks()
        w, h = 860, 680
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        x = (sw - w) // 2
        y = (sh - h) // 2
        self.root.geometry(f"{w}x{h}+{x}+{y}")

        self._build_ui()
        self._start_check()

    # ---------- UI 构建 ----------

    def _build_ui(self):
        """构建界面."""
        # -- 顶部标题 --
        self.header_frame = tk.Frame(self.root, bg=COLORS["bg2"], height=80)
        self.header_frame.pack(fill=tk.X)
        self.header_frame.pack_propagate(False)

        self.title_label = tk.Label(
            self.header_frame,
            text="⏳ 正在检测环境...",
            font=FONT_TITLE, fg=COLORS["fg"], bg=COLORS["bg2"],
        )
        self.title_label.pack(pady=(16, 4))

        self.subtitle_label = tk.Label(
            self.header_frame,
            text="",
            font=FONT_BODY, fg=COLORS["fg_dim"], bg=COLORS["bg2"],
        )
        self.subtitle_label.pack()

        # -- 中间：TreeView --
        self.tree_frame = tk.Frame(self.root, bg=COLORS["bg"])
        self.tree_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=(8, 4))

        columns = ("status", "category", "item", "expected", "actual")
        self.tree = ttk.Treeview(
            self.tree_frame, columns=columns, show="headings",
            height=16,
        )
        self.tree.heading("status", text="")
        self.tree.heading("category", text="类别")
        self.tree.heading("item", text="检测项")
        self.tree.heading("expected", text="期望")
        self.tree.heading("actual", text="实际")

        self.tree.column("status", width=36, anchor=tk.CENTER, stretch=False)
        self.tree.column("category", width=90, anchor=tk.W)
        self.tree.column("item", width=160, anchor=tk.W)
        self.tree.column("expected", width=200, anchor=tk.W)
        self.tree.column("actual", width=220, anchor=tk.W)

        # 滚动条
        vsb = ttk.Scrollbar(self.tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        # -- 底部按钮区 --
        self.btn_frame = tk.Frame(self.root, bg=COLORS["bg"], height=100)
        self.btn_frame.pack(fill=tk.X, padx=12, pady=(4, 12))
        self.btn_frame.pack_propagate(False)

        # 加载中的提示
        self.loading_label = tk.Label(
            self.btn_frame,
            text="正在收集环境信息，请稍候...",
            font=FONT_BODY, fg=COLORS["fg_dim"], bg=COLORS["bg"],
        )
        self.loading_label.pack(pady=20)

    def _rebuild_buttons(self, has_failures: bool, has_env_issue: bool):
        """根据检测结果构建操作按钮."""
        for w in self.btn_frame.winfo_children():
            w.destroy()

        btn_container = tk.Frame(self.btn_frame, bg=COLORS["bg"])
        btn_container.pack(expand=True)

        if has_env_issue:
            # Conda 环境有问题 — 提供创建/选择环境按钮
            tk.Label(
                btn_container,
                text="未找到 quanformer 环境，请选择：",
                font=FONT_BODY, fg=COLORS["warn"], bg=COLORS["bg"],
            ).pack(side=tk.TOP, pady=(8, 4))

            env_btn_frame = tk.Frame(btn_container, bg=COLORS["bg"])
            env_btn_frame.pack(side=tk.TOP, pady=4)

            self._make_btn(env_btn_frame, "创建 quanformer 环境", self._on_create_env,
                           accent=True).pack(side=tk.LEFT, padx=4)

            self._make_btn(env_btn_frame, "使用已有环境...", self._on_select_env).pack(
                side=tk.LEFT, padx=4)

            self._make_btn(env_btn_frame, "跳过 (在当前环境检测)", self._on_skip_env).pack(
                side=tk.LEFT, padx=4)

        else:
            # 环境 OK，展示状态和操作按钮
            summary_text = ""
            if has_failures:
                summary_text = "⚠️ 存在依赖问题，可一键修复 (pip 包)"
            else:
                summary_text = "✅ 环境完全就绪！可直接运行项目"

            tk.Label(
                btn_container,
                text=summary_text,
                font=FONT_BODY, fg=COLORS["fg"], bg=COLORS["bg"],
            ).pack(side=tk.TOP, pady=(8, 2))

            action_btn_frame = tk.Frame(btn_container, bg=COLORS["bg"])
            action_btn_frame.pack(side=tk.TOP, pady=4)

            if has_failures:
                self._make_btn(action_btn_frame, "一键修复", self._on_fix,
                               accent=True).pack(side=tk.LEFT, padx=4)
                self._make_btn(action_btn_frame, "查看修复方案 (不执行)", self._on_fix_dry).pack(
                    side=tk.LEFT, padx=4)

            self._make_btn(action_btn_frame, "重新检测", self._on_recheck).pack(
                side=tk.LEFT, padx=4)
            self._make_btn(action_btn_frame, "重新检测 (其他环境)", self._on_select_env).pack(
                side=tk.LEFT, padx=4)

        # 关闭按钮
        self._make_btn(btn_container, "关闭", self.root.destroy,
                       secondary=True).pack(side=tk.BOTTOM, pady=(8, 0))

    def _make_btn(self, parent, text: str, command, accent=False, secondary=False):
        """创建统一样式的按钮."""
        if secondary:
            bg = COLORS["bg2"]
            fg = COLORS["fg_dim"]
        elif accent:
            bg = COLORS["accent"]
            fg = "#ffffff"
        else:
            bg = COLORS["btn_bg"]
            fg = COLORS["btn_fg"]

        btn = tk.Button(
            parent, text=text, command=command,
            font=FONT_BTN, bg=bg, fg=fg,
            activebackground=COLORS["btn_hover"], activeforeground=COLORS["fg"],
            relief=tk.FLAT, padx=16, pady=6, cursor="hand2",
            borderwidth=0,
        )
        return btn

    # ---------- 检测逻辑 ----------

    def _start_check(self, env_name: Optional[str] = None):
        """后台线程运行检测."""
        self.tree.delete(*self.tree.get_children())
        self.title_label.config(text="⏳ 正在检测环境...")
        self.subtitle_label.config(text="")

        t = threading.Thread(target=self._run_check_thread, args=(env_name,), daemon=True)
        t.start()

    def _run_check_thread(self, env_name: Optional[str] = None):
        """后台执行检测."""
        result = run_check(env_name)
        self.check_result = result
        self.target_env = env_name or os.environ.get("CONDA_DEFAULT_ENV", "")
        self.root.after(0, self._on_check_complete)

    def _on_check_complete(self):
        """检测完成后的 UI 更新."""
        result = self.check_result
        if result is None:
            self.title_label.config(text="❌ 检测失败")
            self.subtitle_label.config(text="无法运行 check_env.py，请检查 Python 环境")
            return

        summary = result.get("summary", {})
        total = summary.get("total", 0)
        passed = summary.get("passed", 0)
        failed_r = summary.get("failed_required", 0)
        failed_o = summary.get("failed_optional", 0)

        if failed_r == 0 and failed_o == 0:
            self.title_label.config(text="✅ 环境依赖全部通过", fg=COLORS["pass"])
        elif failed_r == 0:
            self.title_label.config(
                text=f"⚠️ {passed}/{total} 通过 ({failed_o} 项可选依赖未满足)",
                fg=COLORS["warn"],
            )
        else:
            self.title_label.config(
                text=f"❌ {passed}/{total} 通过 ({failed_r} 项强制失败)",
                fg=COLORS["fail"],
            )

        env_info = f"环境: {self.target_env or '当前 Python'}  |  "
        env_info += f"Python: {result.get('python', 'N/A').split(chr(92))[-1] if result.get('python') else 'N/A'}"
        self.subtitle_label.config(text=env_info)

        # 填充 TreeView
        self.tree.delete(*self.tree.get_children())
        # 配置 tag 样式
        self.tree.tag_configure("pass", foreground=COLORS["pass"])
        self.tree.tag_configure("fail", foreground=COLORS["fail"])
        self.tree.tag_configure("warn", foreground=COLORS["warn"])
        self.tree.tag_configure("even", background=COLORS["row_even"])
        self.tree.tag_configure("odd", background=COLORS["row_odd"])

        has_env_issue = False
        for i, item in enumerate(result.get("results", [])):
            passed_item = item.get("passed", True)
            required = item.get("required", True)
            category = item.get("category", "")
            name = item.get("item", "")
            expected = item.get("expected", "")
            actual = item.get("actual", "")

            if passed_item:
                status_icon = "✅"
                tag = "pass"
            elif not required:
                status_icon = "⚠️"
                tag = "warn"
            else:
                status_icon = "❌"
                tag = "fail"

            row_tag = ("even" if i % 2 == 0 else "odd", tag)
            self.tree.insert(
                "", tk.END,
                values=(status_icon, category, name, expected, actual),
                tags=row_tag,
            )

            # 检测 conda 环境问题
            if category == "Conda" and name == "当前环境" and not passed_item:
                has_env_issue = True

        has_failures = failed_r > 0

        # 重建按钮
        self._rebuild_buttons(has_failures, has_env_issue)

    # ---------- 按钮回调 ----------

    def _on_create_env(self):
        """创建 quanformer conda 环境."""
        if not messagebox.askyesno(
            "创建环境",
            f"将创建名为 '{DEFAULT_ENV_NAME}' 的 conda 环境 (Python 3.11)。\n\n继续？",
        ):
            return

        self.title_label.config(text="⏳ 正在创建 conda 环境...")
        self.root.update()

        def _create():
            ok = create_conda_env(DEFAULT_ENV_NAME)
            self.root.after(0, lambda: self._after_create(ok))

        threading.Thread(target=_create, daemon=True).start()

    def _after_create(self, ok: bool):
        if ok:
            messagebox.showinfo("成功", f"环境 '{DEFAULT_ENV_NAME}' 创建完成！\n将重新检测...")
            self._start_check(DEFAULT_ENV_NAME)
        else:
            messagebox.showerror("失败", f"无法创建环境 '{DEFAULT_ENV_NAME}'。\n请检查 conda 是否已安装。")
            self._start_check()

    def _on_select_env(self):
        """用户选择已有 conda 环境."""
        envs = list_conda_envs()
        if not envs:
            messagebox.showinfo("提示", "未找到其他 conda 环境。\n请先创建 quanformer 环境。")
            return

        dialog = EnvSelectDialog(self.root, envs)
        self.root.wait_window(dialog.top)
        if dialog.result:
            self._start_check(dialog.result)
        else:
            # 用户取消
            pass

    def _on_skip_env(self):
        """跳过 conda 环境，直接在当前环境检测."""
        self._start_check(None)

    def _on_fix(self):
        """一键修复."""
        if not self.target_env:
            messagebox.showwarning("提示", "请先选择或创建 conda 环境。")
            return

        # 收集修复清单
        if not self.check_result:
            return
        fixable = [
            r for r in self.check_result.get("results", [])
            if not r.get("passed") and r.get("fix") and r.get("category") not in
            ("Python", "R", "文件", "磁盘", "硬件", "Conda")
        ]
        if not fixable:
            messagebox.showinfo("提示", "没有可自动修复的项。")
            return

        # 确认修复清单
        lines = ["将安装/更新以下包：", ""]
        for item in fixable:
            lines.append(f"  • {item['item']}: {item['fix']}")
        msg = "\n".join(lines)

        if not messagebox.askyesno("确认修复", msg):
            return

        self.title_label.config(text="⏳ 正在修复环境...")
        self.root.update()

        def _fix():
            output = run_fix(self.target_env, dry_run=False)
            self.fix_output = output
            self.root.after(0, self._after_fix)

        threading.Thread(target=_fix, daemon=True).start()

    def _after_fix(self):
        messagebox.showinfo("修复完成", self.fix_output[:1000] or "修复完成")
        # 重新检测
        self._start_check(self.target_env)

    def _on_fix_dry(self):
        """查看修复方案（不执行）."""
        if not self.target_env:
            messagebox.showwarning("提示", "请先选择或创建 conda 环境。")
            return

        if not self.check_result:
            return
        fixable = [
            r for r in self.check_result.get("results", [])
            if not r.get("passed") and r.get("fix") and r.get("category") not in
            ("Python", "R", "文件", "磁盘", "硬件", "Conda")
        ]
        if not fixable:
            messagebox.showinfo("提示", "没有需要修复的项。")
            return

        lines = ["以下是将执行的修复命令：", ""]
        for item in fixable:
            lines.append(f"  $ {item['fix']}")
        lines.append("")
        lines.append("（以上仅为预览，未实际执行）")
        messagebox.showinfo("修复预览", "\n".join(lines))

    def _on_recheck(self):
        """重新检测."""
        self._start_check(self.target_env if self.target_env else None)


# ---------------------------------------------------------------------------
# 环境选择对话框
# ---------------------------------------------------------------------------

class EnvSelectDialog:
    """让用户从已有 conda 环境列表中选择一个."""

    def __init__(self, parent: tk.Tk, envs: List[str]):
        self.result: Optional[str] = None
        self.top = tk.Toplevel(parent)
        self.top.title("选择 Conda 环境")
        self.top.geometry("400x320")
        self.top.configure(bg=COLORS["bg"])
        self.top.transient(parent)
        self.top.grab_set()

        # 居中
        self.top.update_idletasks()
        w, h = 400, 320
        sw = self.top.winfo_screenwidth()
        sh = self.top.winfo_screenheight()
        self.top.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

        tk.Label(
            self.top, text="选择一个已有的 conda 环境进行检测：",
            font=FONT_HEADING, fg=COLORS["fg"], bg=COLORS["bg"],
            wraplength=360, justify=tk.LEFT,
        ).pack(pady=(16, 8), padx=20)

        # 列表
        list_frame = tk.Frame(self.top, bg=COLORS["bg"])
        list_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=4)

        self.listbox = tk.Listbox(
            list_frame, font=FONT_BODY,
            bg=COLORS["bg2"], fg=COLORS["fg"],
            selectbackground=COLORS["accent"], selectforeground="#ffffff",
            relief=tk.FLAT, borderwidth=0, highlightthickness=0,
        )
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        for env in sorted(envs):
            self.listbox.insert(tk.END, env)
        if envs:
            self.listbox.selection_set(0)

        # 也支持手动输入
        tk.Label(
            self.top, text="或手动输入环境名：",
            font=FONT_BODY, fg=COLORS["fg_dim"], bg=COLORS["bg"],
        ).pack(pady=(8, 0), padx=20)

        self.entry_var = tk.StringVar()
        self.entry = tk.Entry(
            self.top, textvariable=self.entry_var, font=FONT_BODY,
            bg=COLORS["bg2"], fg=COLORS["fg"], insertbackground=COLORS["fg"],
            relief=tk.FLAT, borderwidth=0,
        )
        self.entry.pack(fill=tk.X, padx=20, pady=4, ipady=4)

        # 按钮
        btn_frame = tk.Frame(self.top, bg=COLORS["bg"])
        btn_frame.pack(pady=(8, 16))

        tk.Button(
            btn_frame, text="确定", command=self._on_confirm,
            font=FONT_BTN, bg=COLORS["accent"], fg="#ffffff",
            relief=tk.FLAT, padx=20, pady=4, cursor="hand2",
        ).pack(side=tk.LEFT, padx=4)

        tk.Button(
            btn_frame, text="取消", command=self.top.destroy,
            font=FONT_BTN, bg=COLORS["bg2"], fg=COLORS["fg_dim"],
            relief=tk.FLAT, padx=20, pady=4, cursor="hand2",
        ).pack(side=tk.LEFT, padx=4)

        self.listbox.bind("<Double-Button-1>", lambda e: self._on_confirm())

    def _on_confirm(self):
        """确认选择."""
        # 优先手动输入
        manual = self.entry_var.get().strip()
        if manual:
            self.result = manual
        else:
            sel = self.listbox.curselection()
            if sel:
                self.result = self.listbox.get(sel[0])
            else:
                messagebox.showwarning("提示", "请选择一个环境或输入环境名")
                return
        self.top.destroy()


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main():
    root = tk.Tk()
    try:
        root.iconbitmap(default="")
    except Exception:
        pass
    CheckGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
