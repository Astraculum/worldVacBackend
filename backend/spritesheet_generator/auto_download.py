import asyncio
import atexit
import os
import signal
import socket
import subprocess
import time
from contextlib import closing

from playwright.sync_api import TimeoutError, sync_playwright


class CharacterImageDownloader:
    MAX_RETRY = 3

    def __init__(
        self,
        max_concurrent_downloads: int = 30,
    ):
        self.base_url = ""
        self.semaphore = asyncio.Semaphore(max_concurrent_downloads)

    def find_free_port(
        self,
    ):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(("", 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return s.getsockname()[1]

    def close(self):
        if not self._closed:
            self.process.send_signal(sig=signal.SIGTERM)
            self.process.wait()
            self._closed = True

    def start_character_generation_server(
        self, index_html_path: str = "Universal-LPC-Spritesheet-Character-Generator/"
    ):
        self.port = self.find_free_port()
        cmd = f"python -m http.server {self.port} --directory {index_html_path} --bind 0.0.0.0"
        self.process = subprocess.Popen(
            args=cmd,
        )
        self.base_url = f"http://localhost:{self.port}"
        time.sleep(5)
        atexit.register(self.close)

    async def download_character_image(
        self,
        params: str = "",
        output_dir: str = "generated",
        output_filename: str = "character.png",
        overwrite: bool = False,
    ):
        """
        自动打开角色生成器页面并下载生成的图片
        """
        if not self.base_url:
            raise ValueError(
                "Character generation server not started, please start it with `start_character_generation_server`"
            )
        if params:
            url = f"{self.base_url.rstrip('/')}/#?{params}"
        else:
            url = self.base_url
        output_path = os.path.join(output_dir, output_filename)
        if not overwrite and os.path.exists(output_path):
            return
        async with self.semaphore:
            for i in range(self.MAX_RETRY):
                with sync_playwright() as p:
                    # 启动浏览器，设置更长的超时时间
                    browser = p.chromium.launch(
                        headless=True, args=["--no-sandbox", "--disable-setuid-sandbox"]
                    )

                    # 设置下载行为
                    context = browser.new_context(
                        viewport={"width": 1920, "height": 1080},
                        accept_downloads=True,  # 启用下载
                    )

                    page = context.new_page()

                    try:
                        # 访问页面
                        # print(f"正在访问页面: {url}")
                        page.goto(url, timeout=60000)

                        # 等待页面加载完成
                        # print("等待页面加载...")
                        page.wait_for_selector("#saveAsPNG", timeout=60000)

                        # 等待一段时间，确保角色生成完成
                        # print("等待角色生成...")
                        time.sleep(5)

                        # 创建一个下载监听器
                        with page.expect_download() as download_info:
                            # 点击下载按钮
                            # print("点击下载按钮...")
                            page.click("#saveAsPNG")

                            # 等待下载开始
                            download = download_info.value

                            # 保存文件
                            os.makedirs(output_dir, exist_ok=True)

                            # 保存文件
                            download.save_as(output_path)
                            # print(f"图片已保存到: {output_path}")
                            break

                    except TimeoutError as e:
                        pass
                        # print(f"页面加载超时: {str(e)}")
                        # print("请确保服务器正在运行，并且可以访问")
                    except Exception as e:
                        # print(f"发生错误: {str(e)}")
                        # 保存页面截图以便调试
                        # try:
                        #     debug_dir = os.path.join(os.path.dirname(__file__), "debug")
                        #     os.makedirs(debug_dir, exist_ok=True)
                        #     screenshot_path = os.path.join(
                        #         debug_dir, f"error_{int(time.time())}.png"
                        #     )
                        #     page.screenshot(path=screenshot_path)
                        #     print(f"错误截图已保存到: {screenshot_path}")
                        # except:
                        #     print("无法保存错误截图")
                        pass
                    finally:
                        browser.close()
