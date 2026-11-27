CHALLENGE_TITLES = [
    'Just a moment...',
    '请稍候…',
    'DDOS-GUARD',
]

CHALLENGE_SELECTORS = [
    '#cf-challenge-running', '.ray_id', '.attack-box', '#cf-please-wait',
    '#challenge-spinner', '#trk_jschal_js', 'td.info #js_info', 'div.vc div.text-box h2'
]


def resolve_challenge(driver=None, timeout=60):
    """
    轻量空壳实现：已移除 Selenium / Chromium 浏览器仿真组件
    """
    return False


def under_challenge(html_text: str):
    """
    检查返回 HTML 是否处于 Cloudflare / DDoS-Guard 质询页面（纯文本/DOM 检测，不依赖浏览器）
    """
    if not html_text:
        return False
    try:
        from pyquery import PyQuery
        html_doc = PyQuery(html_text)
        page_title = (html_doc('title').text() or "").strip().lower()
        for title in CHALLENGE_TITLES:
            if page_title == title.lower():
                return True
        for selector in CHALLENGE_SELECTORS:
            if html_doc(selector):
                return True
    except Exception:
        pass
    return False
