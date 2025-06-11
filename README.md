# World Vac 后端

## 初始化

0. 安装uv
```bash
curl -fsSL https://get.uv.dev | sh
```

1. 安装依赖
```bash
uv sync
```

2. 初始化submodules
```bash
git submodule update --init --recursive
```

3. 初始化node
```bash
cd worldVacBackend/AgentMatrix/FMG
npm install 
yum install -y atk at-spi2-atk cups libXcomposite libXdamage libXrandr libgbm alsa-lib pango
```

4. 运行后端
```bash
python3 main.py
```
