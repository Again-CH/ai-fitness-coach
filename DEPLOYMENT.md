# AI 健身教练 - Docker 部署指南

## 目录结构

```
ai-fitness-coach/
├── backend/              # 后端代码（FastAPI）
│   ├── main.py
│   ├── database.py
│   ├── langgraph_brain.py
│   ├── multimodal_service.py
│   ├── static/          # 前端静态文件
│   │   └── index.html
│   ├── fitness.db       # SQLite 数据库（自动创建）
│   └── .env            # 环境变量配置
├── Dockerfile           # Docker 镜像构建文件
├── docker-compose.yml   # Docker Compose 编排文件
├── requirements.txt     # Python 依赖列表
└── .dockerignore       # Docker 构建排除文件
```

## 前置条件

1. 安装 Docker Desktop（Windows/Mac）或 Docker Engine（Linux）
2. 确保 Docker 版本 >= 20.10.0
3. 确保 Docker Compose 版本 >= 1.29.0

## 部署步骤

### 1. 配置环境变量

在 `backend/.env` 文件中配置以下环境变量：

```env
# 通义千问 API Key（必填）
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxx

# 通义千问模型（可选，默认 qwen-plus）
DASHSCOPE_MODEL=qwen-plus

# 腾讯云配置（可选，用于 ASR 和 TTS）
TENCENT_SECRET_ID=your_secret_id
TENCENT_SECRET_KEY=your_secret_key
TENCENT_APP_ID=your_app_id

# TTS 音色（可选，默认 101014=晓浩-活力男声）
TENCENT_TTS_VOICE_TYPE=101014

# JWT 密钥（可选，默认使用内置密钥）
JWT_SECRET_KEY=your-secret-key-change-in-production
```

### 2. 构建 Docker 镜像

在项目根目录下运行：

```bash
docker-compose build
```

这会使用 Dockerfile 构建后端服务的镜像。

### 3. 启动服务

在项目根目录下运行：

```bash
docker-compose up -d
```

这会在后台启动后端服务。

### 4. 查看服务状态

```bash
docker-compose ps
```

### 5. 查看服务日志

```bash
docker-compose logs -f backend
```

### 6. 停止服务

```bash
docker-compose down
```

## 访问应用

服务启动后，可以通过以下 URL 访问：

- **前端页面**：http://localhost:8000/
- **后端 API 文档**：http://localhost:8000/docs
- **健康检查**：http://localhost:8000/health

## 数据持久化

SQLite 数据库文件（`backend/fitness.db`）会通过 Docker 卷挂载，所以即使容器重启，数据也不会丢失。

如果需要备份数据库，可以直接复制 `backend/fitness.db` 文件。

## 更新部署

如果代码有更新，需要重新构建镜像并重启服务：

```bash
docker-compose down
docker-compose build
docker-compose up -d
```

## 生产环境部署

### 1. 修改 JWT 密钥

在 `backend/.env` 文件中设置强随机的 JWT 密钥：

```env
JWT_SECRET_KEY=your-super-secret-key-at-least-32-characters-long
```

可以使用以下命令生成随机密钥：

```bash
openssl rand -hex 32
```

### 2. 启用 HTTPS

在生产环境中，建议使用 Nginx 或 Caddy 作为反向代理，并配置 SSL 证书。

示例 Nginx 配置：

```nginx
server {
    listen 443 ssl;
    server_name your-domain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 3. 使用云服务器部署

1. 在云服务器（如阿里云、腾讯云、AWS）上安装 Docker 和 Docker Compose
2. 将代码上传到服务器（可以使用 Git 或 SCP）
3. 配置防火墙，开放 80/443 端口
4. 按照上述步骤启动服务
5. 配置域名解析到服务器公网 IP

## 常见问题

### 1. 端口冲突

如果 8000 端口已经被占用，可以修改 `docker-compose.yml` 中的端口映射：

```yaml
ports:
  - "8080:8000"  # 将主机端口改为 8080
```

### 2. 数据库文件权限问题

如果遇到数据库文件权限问题，可以修改 `docker-compose.yml`，添加用户 ID 映射：

```yaml
environment:
  - PUID=1000
  - PGID=1000
```

### 3. 腾讯云 API 无法访问

如果遇到腾讯云 API 无法访问的问题，请检查：
1. `backend/.env` 文件中的配置是否正确
2. 容器是否可以访问外网（可以使用 `docker exec -it ai-fitness-coach-backend ping api.tencentcloud.com` 测试）

## 卸载

如果需要卸载，可以运行：

```bash
docker-compose down -v  # -v 参数会删除挂载的卷
docker rmi ai-fitness-coach-backend  # 删除镜像
```

---

**完成！🎉** 现在你可以使用 Docker 快速部署 AI 健身教练应用了！
