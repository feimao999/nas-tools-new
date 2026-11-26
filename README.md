# NAS媒体库管理工具（个人自用精简定制版）

[![GitHub release](https://img.shields.io/github/v/release/feimao999/nas-tools-new?style=plastic)](https://github.com/feimao999/nas-tools-new/releases)
[![GitHub stars](https://img.shields.io/github/stars/feimao999/nas-tools-new?style=plastic)](https://github.com/feimao999/nas-tools-new/stargazers)
[![GitHub license](https://img.shields.io/github/license/feimao999/nas-tools-new?style=plastic)](https://github.com/feimao999/nas-tools-new/blob/master/LICENSE.md)
[![Platform](https://img.shields.io/badge/platform-amd64%2Farm64-blue?style=plastic)](https://github.com/feimao999/nas-tools-new/pkgs/container/nas-tools-new)

## 维护声明

1. **个人自用声明**：本仓库仅为**个人自用**而维护的定制精简版本，不建任何交流群或频道，也不承诺对外提供技术支持与长期兼容维护。
2. **AI 编写声明**：本仓库在原项目基础上所做的**全部新增功能、重构与体积精简改动，均 100% 由 AI 编写完成**。
3. **项目来源与致谢**：本项目直接 Fork 并修改自 [c77187326/nas-tools](https://github.com/c77187326/nas-tools)（上游源自 [hsuyelin/nas-tools](https://github.com/hsuyelin/nas-tools) 及官方原版 [NAStool/nas-tools](https://github.com/NAStool/nas-tools)），特此感谢原作者与历任开源维护者的付出。
4. **开源与低调使用**：所有修改代码均保持开源供参考；请自行低调使用，勿在公共交流群或频道大肆传播。

---

## 本仓库定制与精简说明（v4.0+）

围绕「**资源搜索 + 追更/RSS 订阅 + 下载管理 + AI 智能识别整理**」核心场景进行了重构与大幅瘦身：

### 1. 多模型联网 AI 识别增强
- [x] 支持 **OpenAI**、**Anthropic Claude**、**Google Gemini** 及 **本地/第三方兼容接口（Ollama、LM Studio、DeepSeek、One-API 等）** 原生直连，支持一键拉取模型列表。
- [x] 支持 **优先使用 AI 识别**（遇到额度耗尽或网络异常自动回退常规 TMDB 识别，不中断整理流程）。
- [x] 支持 **AI 联网搜索辅助识别**，解决新剧场版、冷门番剧、罗马音/副标题命名识别不准问题。
- [x] 支持两阶段 TMDB 候选精准选片、多季不同标题动画季号缓存，以及 `OVA` / `OAD` / `SP` 特典自动归入 `Season 0 (S00)`。

### 2. 镜像体积与运行占用大精简（镜像体积缩减约 66%）
- [x] **移除重型底层组件**：移除内置无头浏览器 `Chromium`、`Xvfb`、`Selenium`、`ffmpeg`、`MinIO mc` 及 20+ 个冗余 Python 包（保留 `rclone` 与 `Redis + feapder` 搜索支持）。
- [x] **精简无关界面与后台任务**：移除 `探索`、`站点管理`（站点维护/数据统计/刷流任务/站点资源）、顶部探索搜索栏、基础设置 `OCR` 卡片及对应后台定时任务。
- [x] **清理无用插件**：移除 `站点自动签到`、`AI字幕自动生成`、`IYUU自动辅种`、`CookieCloud同步`、`自动转移做种`、`种子标记`、`OpenSubtitles` 插件。
- [x] 详细每次变更说明请查阅 [GitHub Releases](https://github.com/feimao999/nas-tools-new/releases)。

---

## 安装

### 1、Docker（推荐，支持 `linux/amd64` 与 `linux/arm64`）
```bash
docker pull ghcr.io/feimao999/nas-tools-new:latest
```
也可指定具体版本号拉取（详见 [Releases](https://github.com/feimao999/nas-tools-new/releases)）：
```bash
docker pull ghcr.io/feimao999/nas-tools-new:v4.0.0
```
Docker Compose 示例可参考仓库内 [`docker/compose.yml`](docker/compose.yml)。

### 2、本地源码运行
需要 Python 3.10 环境并预装 `Cython`（`python3 -m pip install Cython`）：
```bash
git clone -b master https://github.com/feimao999/nas-tools-new.git --recurse-submodule
cd nas-tools-new
python3 -m pip install -r requirements.txt
export NASTOOL_CONFIG="/xxx/config/config.yaml"
nohup python3 run.py &
```

---

## 免责声明

1. 本软件仅供个人学习交流使用，对使用者的行为及内容概不知情，使用本软件产生的任何责任均由使用者本人自行承担。
2. 本项目没有任何捐赠渠道，也绝不会接受捐赠或收费，请注意甄别。

---

## 常见问题

### 1. 启动 inotify 报错 / 无法自动目录同步
> 日志报 `inotify instance limit reached`、`inotify watch limit reached` 等错误，或目录同步无法自动触发。

解决办法（在宿主机而非容器内执行）：
```bash
echo fs.inotify.max_user_watches=524288 | sudo tee -a /etc/sysctl.conf
echo fs.inotify.max_user_instances=524288 | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```
或者启用内置插件「定时目录同步」。

### 2. 启动报错数据库 `no such column`
使用 [SQLite 浏览器](https://github.com/sqlitebrowser/sqlitebrowser) 打开 `config` 目录下的 `user.db`，删除 `alembic_version` 表后重启容器。

### 3. 消息通知内容点击无法跳转
在「系统设置 -> 基础设置 -> 系统 -> 外网访问地址」中填写访问地址。

### 4. 电影/电视剧订阅一直处于队列中
- 如启用了订阅站点，请在「系统设置 -> 基础设置 -> 服务 -> 订阅RSS周期」中设置周期；
- 如启用了搜索站点，请在「订阅搜索周期」中设置周期，或手动点击订阅卡片刷新。

### 5. 目录同步文件重复转移
目录同步设置时，**目的目录不要放在源目录内部**。

### 6. 识别转移错误码 `-1`
硬链接不支持跨文件系统/跨盘（群晖中不同共享文件夹会被视为跨卷），请确保源目录与目标媒体库挂载在同一个父级挂载路径下。

### 7. 电视剧订阅在完结前自动删除
TMDB 词条尚未更新总集数时，可在订阅编辑中手动指定总集数。
