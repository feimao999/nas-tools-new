FROM python:3.10.11-alpine
COPY package_list.txt requirements.txt /tmp/
RUN apk add --no-cache --virtual .build-deps \
        libffi-dev \
        gcc \
        musl-dev \
        libxml2-dev \
        libxslt-dev \
    && apk add --no-cache $(echo $(cat /tmp/package_list.txt)) \
    && curl https://rclone.org/install.sh | bash \
    && echo "pip==23.3.2" > /tmp/pip-constraints.txt \
    && echo "setuptools==68.2.2" >> /tmp/pip-constraints.txt \
    && echo "wheel==0.41.2" >> /tmp/pip-constraints.txt \
    && echo "cython==0.29.36" >> /tmp/pip-constraints.txt \
    && export PIP_CONSTRAINT=/tmp/pip-constraints.txt \
    && pip install --upgrade "pip==23.3.2" "setuptools==68.2.2" "wheel==0.41.2" "cython==0.29.36" \
    && pip install --no-build-isolation -r /tmp/requirements.txt \
    && apk del --purge .build-deps \
    && rm -rf /tmp/* /root/.cache /var/cache/apk/*
ENV S6_SERVICES_GRACETIME=30000 \
    S6_KILL_GRACETIME=60000 \
    S6_CMD_WAIT_FOR_SERVICES_MAXTIME=0 \
    S6_SYNC_DISKS=1 \
    HOME="/nt" \
    TERM="xterm" \
    LANG="C.UTF-8" \
    TZ="Asia/Shanghai" \
    NASTOOL_CONFIG="/config/config.yaml" \
    NASTOOL_AUTO_UPDATE=false \
    NASTOOL_CN_UPDATE=true \
    NASTOOL_VERSION=master \
    PS1="\u@\h:\w \$ " \
    REPO_URL="https://github.com/feimao999/nas-tools-new.git" \
    PYPI_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple" \
    ALPINE_MIRROR="mirrors.ustc.edu.cn" \
    PUID=0 \
    PGID=0 \
    UMASK=000 \
    PYTHONWARNINGS="ignore:semaphore_tracker:UserWarning" \
    WORKDIR="/nas-tools"
WORKDIR ${WORKDIR}
ARG BUILD_COMMIT=unknown
ARG BUILD_VERSION=""
ENV NASTOOL_IMAGE_VERSION=${BUILD_VERSION}
RUN echo "Building nas-tools-new commit ${BUILD_COMMIT} (${BUILD_VERSION})"
RUN mkdir ${HOME} \
    && addgroup -S nt -g 911 \
    && adduser -S nt -G nt -h ${HOME} -s /bin/bash -u 911 \
    && python_ver=$(python3 -V | awk '{print $2}') \
    && python_path=$(which python3) \
    && [ -d "/usr/lib/python${python_ver%.*}/site-packages" ] || mkdir -p "/usr/lib/python${python_ver%.*}/site-packages" \
    && echo "${WORKDIR}/" > /usr/lib/python${python_ver%.*}/site-packages/nas-tools.pth \
    && echo 'fs.inotify.max_user_watches=5242880' >> /etc/sysctl.conf \
    && echo 'fs.inotify.max_user_instances=5242880' >> /etc/sysctl.conf \
    && echo "nt ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers \
    && git config --global pull.ff only \
    && git clone -b master ${REPO_URL} ${WORKDIR} --depth=1 --recurse-submodule \
    && git config --global --add safe.directory ${WORKDIR}
COPY --chmod=755 ./docker/rootfs /
EXPOSE 3000
VOLUME ["/config"]
ENTRYPOINT [ "/init" ]
