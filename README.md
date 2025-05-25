# 智能驾驶舱系统

## 环境配置

### 系统要求

- Python 3.12
- CUDA (GPU加速，可选)
- Windows 10/11

### 安装步骤

1. **克隆仓库**

```bash
   git clone https://github.com/CentaureaHO/NKU_SoftwareEngineering.git
   cd NKU_SoftwareEngineering
```

2. **创建并激活conda环境**

```bash
   conda create -n smart_cab python=3.12
   conda activate smart_cab
```

3. **安装依赖**

```bash
   pip install -r requirements.txt
```

4. **安装PyTorch**

   使用GPU加速(CUDA 12.6):

```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

   或使用CPU版本:

```bash
   pip install torch torchvision torchaudio
```

5. **安装ModelScope**

   仅需要音频部分，使用：

```bash
   pip install modelscope[audio] -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
```
6. **安装Kokoro Zh**

   用于语音合成的Kokoro库的中文语音合成需额外安装：

```bash
   pip install misaki[zh]
```