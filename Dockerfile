# 使用 Miniconda 作为基础镜像
FROM continuumio/miniconda3

# 设置工作目录
WORKDIR /app

# 复制文件到 Docker 镜像（dataset/ 会被 .dockerignore 忽略）
COPY . /app

# 创建 Conda 环境
RUN conda env create -f environment.yml

# 设置默认 shell 使用 conda 环境
SHELL ["conda", "run", "-n", "codetr", "/bin/bash", "-c"]

# （可选）默认执行 bash
CMD ["bash"]