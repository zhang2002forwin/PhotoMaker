"""从 freshluts.com 批量下载 LUT 文件到 ./luts 目录。

网站结构:
  - 列表页: https://freshluts.com/luts?page=N  (每页若干个 LUT 卡片)
  - 详情页: https://freshluts.com/luts/{id}    (含下载按钮)
  - 下载方式: POST /downloadlut?lutid={lut_id}&userid={user_id}
              表单字段: authenticity_token (从详情页提取)
  - 下载需登录: 未登录时下载按钮指向 /users/sign_up

使用方式:
  1. 通过 Cookie 登录后下载 (推荐):
     python download_luts.py --pages 1-10 --cookie "_starter_session=xxxxx"

  2. 通过账号密码登录后下载:
     python download_luts.py --pages 1-10 --email you@example.com --password yourpass

  3. 限制下载数量:
     python download_luts.py --pages 1-50 --max 100

  4. 指定输出目录:
     python download_luts.py --pages 1-10 --out_dir data/luts

获取 Cookie 的方法:
  1. 在浏览器登录 freshluts.com
  2. 按 F12 -> Application -> Cookies -> https://freshluts.com
  3. 找到 _starter_session, 复制它的 Value
  4. 运行: python download_luts.py --cookie "_starter_session=复制的值"

  或者在 Console 运行: copy(document.cookie)

注意:
  - 请遵守网站的 robots.txt 和使用条款
  - 建议设置合理的 --delay 避免请求过于频繁
  - LUT 文件格式通常为 .cube
"""

import os
import re
import sys
import time
import argparse
import getpass
from urllib.parse import urljoin, urlparse, parse_qs

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("缺少依赖，请先安装:")
    print("  pip install requests beautifulsoup4")
    sys.exit(1)


BASE_URL = "https://freshluts.com"
DEFAULT_OUT_DIR = "./luts"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def parse_page_range(s: str):
    """解析页码范围字符串，如 '1-5' 或 '3' 或 '1,3,5'。"""
    pages = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            pages.extend(range(int(a), int(b) + 1))
        elif part:
            pages.append(int(part))
    return sorted(set(pages))


def create_session(cookie: str = None, email: str = None, password: str = None):
    """创建带登录状态的 requests.Session。"""
    session = requests.Session()
    session.headers.update(HEADERS)

    if cookie:
        session.headers["Cookie"] = cookie
        print("已设置 Cookie")

    if email and password:
        print(f"正在登录: {email}")
        resp = session.get(urljoin(BASE_URL, "/users/sign_in"), timeout=15)
        soup = BeautifulSoup(resp.text, "html.parser")
        token_input = soup.find("input", {"name": "authenticity_token"})
        token = token_input["value"] if token_input else ""

        login_data = {
            "utf8": "\u2713",
            "authenticity_token": token,
            "user[email]": email,
            "user[password]": password,
            "user[remember_me]": "1",
            "commit": "Log in",
        }
        resp = session.post(
            urljoin(BASE_URL, "/users/sign_in"),
            data=login_data,
            timeout=15,
            allow_redirects=True,
        )
        if "Invalid" in resp.text or "sign_in" in resp.url:
            print("登录失败: 邮箱或密码错误")
            print("请改用 --cookie 方式登录")
            sys.exit(1)
        else:
            print("登录成功")

    return session


def get_lut_links_from_page(session, page_num: int):
    """从列表页提取所有 LUT 详情页链接。"""
    url = urljoin(BASE_URL, f"/luts?page={page_num}")
    resp = session.get(url, timeout=15)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    links = []
    for a in soup.find_all("a", href=re.compile(r"^/luts/\d+$")):
        href = a["href"]
        full_url = urljoin(BASE_URL, href)
        if full_url not in links:
            links.append(full_url)
    return links


def get_download_form_from_detail(session, detail_url: str):
    """从 LUT 详情页提取下载表单信息。

    登录后下载按钮是 form POST:
        <form class="button_to" method="post" action="/downloadlut?lutid=983&userid=1445141">
            <input type="hidden" name="authenticity_token" value="...">
        </form>

    返回: (action_url, token, lut_name) 或 (None, None, name)
    """
    resp = session.get(detail_url, timeout=15)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # 提取 LUT 名称
    title = ""
    h_tag = soup.find(["h1", "h2", "h3"])
    if h_tag:
        title = h_tag.get_text(strip=True)
    if not title:
        title = detail_url.rstrip("/").split("/")[-1]

    # 查找下载表单: <form action="/downloadlut?lutid=...&userid=...">
    for form in soup.find_all("form"):
        action = form.get("action", "")
        if "/downloadlut" in action:
            token_input = form.find("input", {"name": "authenticity_token"})
            token = token_input["value"] if token_input else ""
            action_url = urljoin(BASE_URL, action)
            return action_url, token, title

    # 未找到下载表单 - 可能未登录
    for a in soup.find_all("a", href=True):
        if "sign_up" in a["href"] and "download" in a.get_text(strip=True).lower():
            print(f"  需要登录才能下载: {title}")
            break

    return None, None, title


def download_file(session, action_url: str, token: str, out_path: str):
    """通过 POST 表单下载 LUT 文件。"""
    resp = session.post(
        action_url,
        data={"authenticity_token": token},
        timeout=60,
        stream=True,
        allow_redirects=True,
    )
    resp.raise_for_status()

    # 从 Content-Disposition 获取文件名
    content_disp = resp.headers.get("Content-Disposition", "")
    if "filename=" in content_disp:
        fname = re.search(r'filename="?([^";\n]+)"?', content_disp)
        if fname:
            ext = os.path.splitext(fname.group(1))[1]
            if ext:
                out_path = os.path.splitext(out_path)[0] + ext

    # 检查响应是否是 HTML (登录失效会返回 HTML 而非文件)
    content_type = resp.headers.get("Content-Type", "")
    if "text/html" in content_type:
        raise RuntimeError("下载失败: 返回了 HTML 而非文件 (Cookie 可能已失效)")

    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

    return out_path


def sanitize_filename(name: str):
    """清理文件名，移除非法字符。"""
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    name = name.strip().strip(".")
    return name[:100] if len(name) > 100 else name


def main():
    parser = argparse.ArgumentParser(
        description="从 freshluts.com 下载 LUT 文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--pages", type=str, default="1-5",
                        help="页码范围，如 '1-5' 或 '1,3,5' (默认 1-5)")
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR,
                        help=f"输出目录 (默认 {DEFAULT_OUT_DIR})")
    parser.add_argument("--cookie", type=str, default=None,
                        help="登录 Cookie 字符串 (推荐)")
    parser.add_argument("--email", type=str, default='z1845404964@gmail.com',
                        help="登录邮箱")
    parser.add_argument("--password", type=str, default='z1845404964',
                        help="登录密码 (不填则交互式输入)")
    parser.add_argument("--max", type=int, default=None,
                        help="最大下载数量")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="每次请求间隔秒数 (默认 1.0)")
    parser.add_argument("--overwrite", action="store_true",
                        help="覆盖已下载的文件")
    args = parser.parse_args()

    if args.email and not args.password:
        args.password = getpass.getpass("密码: ")

    if not args.cookie and not args.email:
        print("=" * 60)
        print("警告: 未提供登录信息")
        print("freshluts.com 需要登录才能下载 LUT 文件。")
        print("请使用 --cookie 或 --email/--password 参数。")
        print("=" * 60)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    pages = parse_page_range(args.pages)
    print(f"将抓取第 {pages[0]}-{pages[-1]} 页 (共 {len(pages)} 页)")
    print(f"输出目录: {os.path.abspath(args.out_dir)}")
    print(f"请求间隔: {args.delay}s")
    if args.max:
        print(f"最大下载数: {args.max}")
    print()

    session = create_session(
        cookie=args.cookie, email=args.email, password=args.password
    )

    # 收集所有 LUT 详情页链接
    all_lut_urls = []
    for page_num in pages:
        print(f"正在抓取列表页 {page_num}...")
        try:
            links = get_lut_links_from_page(session, page_num)
            all_lut_urls.extend(links)
            print(f"  找到 {len(links)} 个 LUT 链接")
        except Exception as e:
            print(f"  抓取失败: {e}")
        time.sleep(args.delay)

    # 去重
    seen = set()
    unique_urls = []
    for url in all_lut_urls:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)

    print(f"\n共发现 {len(unique_urls)} 个不重复的 LUT")
    if args.max:
        unique_urls = unique_urls[: args.max]
        print(f"限制下载前 {args.max} 个")

    # 逐个下载
    success = 0
    failed = 0
    skipped = 0

    for i, detail_url in enumerate(unique_urls, 1):
        print(f"\n[{i}/{len(unique_urls)}] {detail_url}")

        try:
            # 从详情页提取下载表单
            action_url, token, name = get_download_form_from_detail(session, detail_url)

            if not action_url:
                failed += 1
                continue

            # 构造输出路径
            safe_name = sanitize_filename(name) or f"lut_{detail_url.rstrip('/').split('/')[-1]}"
            out_path = os.path.join(args.out_dir, safe_name + ".cube")

            if os.path.exists(out_path) and not args.overwrite:
                print(f"  已存在，跳过: {safe_name}")
                skipped += 1
                continue

            print(f"  下载: {name}")
            saved_path = download_file(session, action_url, token, out_path)
            size_kb = os.path.getsize(saved_path) / 1024
            print(f"  完成: {os.path.basename(saved_path)} ({size_kb:.1f} KB)")
            success += 1

        except Exception as e:
            print(f"  失败: {e}")
            failed += 1

        time.sleep(args.delay)

    # 汇总
    print("\n" + "=" * 60)
    print(f"下载完成:")
    print(f"  成功: {success}")
    print(f"  跳过: {skipped}")
    print(f"  失败: {failed}")
    print(f"  总计: {len(unique_urls)}")
    print(f"  目录: {os.path.abspath(args.out_dir)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
