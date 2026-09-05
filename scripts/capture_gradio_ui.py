"""Fill Gradio demo with an example image and save app_ui.png."""

from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "demo" / "app_ui.png"
URL = "http://127.0.0.1:7861/"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, channel="chrome")
        page = browser.new_page(viewport={"width": 1440, "height": 1280})
        page.goto(URL, wait_until="domcontentloaded", timeout=120000)
        page.get_by_role("button", name="检测并评估风险").wait_for(timeout=120000)
        thumbs = page.locator(".gallery-item, .thumbnail-item, button.thumbnail-item")
        if thumbs.count() == 0:
            thumbs = page.locator("img").nth(0)
            page.locator("text=示例巡检图").locator("xpath=following::img[1]").click(timeout=30000)
        else:
            thumbs.first.click(timeout=30000)
        page.wait_for_timeout(2500)
        page.get_by_role("button", name="检测并评估风险").click()
        page.get_by_text("\u98ce\u9669\u7b49\u7ea7").wait_for(timeout=180000)
        page.wait_for_timeout(1500)
        page.screenshot(path=str(OUT), full_page=True)
        browser.close()
    print("saved", OUT, "bytes", OUT.stat().st_size)


if __name__ == "__main__":
    main()
