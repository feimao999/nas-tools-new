class ChromeHelper(object):
    """
    轻量空壳实现：已移除内置 Chromium / Selenium 浏览器组件以降低体积与内存占用
    """

    def __init__(self, headless=False):
        self._headless = headless

    def init_driver(self):
        return None

    @property
    def browser(self):
        return None

    def get_status(self):
        return False

    def visit(self, url, ua=None, apikey=None, cookie=None, timeout=30, proxy=None):
        return False

    def new_tab(self, url, ua=None, apikey=None, cookie=None):
        return False

    def close_tab(self):
        return False

    def pass_cloudflare(self):
        return False

    def execute_script(self, script):
        return False

    def get_title(self):
        return ""

    def get_html(self):
        return ""

    def get_cookies(self):
        return ""

    def get_ua(self):
        return None

    def quit(self):
        return None


def init_chrome():
    return None
