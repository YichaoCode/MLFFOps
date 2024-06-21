# AutoDPGen 使用文档

## 环境准备

导航到本目录：

```bash
cd /home/yinbc/yichao/dev-002-dpgen-dimer/Cu_COH_dpa2/auto
```

在使用DPGen之前, 需要先激活相应的Python环境：

```bash
conda activate deepmd_yichao
```

## 基本用法

DPGen的主要入口是`main.py`脚本, 位于`auto`目录下。基本用法如下:

```bash
python main.py --clean
```
增加 --debug参数
```bash
python main.py --clean --debug
```

```bash
python main.py par.json machine.json [选项]
```

其中:

- `par.json`: 参数配置文件路径
- `machine.json`: 机器配置文件路径
- `[选项]`: 可选参数, 见下文

## 可选参数

- `--clean`: 在运行前清理文件和文件夹
- `--restart <ITER.TASK>`: 从指定的任务重新开始, 格式为`ITER.TASK`, 例如`000000.06`
- `-d, --debug`: 启用调试模式
- `-v, --version`: 显示版本信息
- `-p, --param <file>`: 指定参数配置文件路径, 默认为`input/par.json`
- `-m, --machine <file>`: 指定机器配置文件路径, 默认为`input/machine.json`

## 示例

### 清理并重新运行

```bash
python main.py par.json machine.json --clean
```

也可以指定配置文件路径：

```bash
python main.py -p input/par.json -m input/machine.json --clean
```

### 从指定任务重新开始

```bash
python main.py par.json machine.json --restart 000003.05
```

## 代码结构

以下是DPGenOpt的代码结构和主要文件说明:

```
DPGenOpt/
├── auto/
│   ├── config/
│   │   ├── args.py                  # 命令行参数解析
│   │   ├── config.py                # 配置管理
│   │   ├── __init__.py
│   │   └── logger.py                # 日志管理
│   ├── fp_interface/
│   │   ├── fp_abacus.py             # ABACUS接口
│   │   ├── fp_cp2k.py               # CP2K接口
│   │   ├── fp_gaussian.py           # Gaussian接口
│   │   ├── fp_pwmat.py              # PWMAT接口
│   │   ├── fp_pwscf.py              # PWSCF接口
│   │   ├── fp_siesta.py             # SIESTA接口
│   │   ├── fp_vasp.py               # VASP接口
│   │   └── __init__.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── modi_json.py             # JSON文件修改
│   │   ├── parse_utils.py           # 解析工具函数
│   │   ├── path_utils.py            # 路径处理工具函数
│   │   ├── utils.py                 # 通用工具函数
│   │   └── vasp_utils.py            # VASP相关工具函数
│   ├── workflow/
│   │   ├── __init__.py
│   │   ├── iter_func.py             # 迭代功能实现
│   │   ├── lammps_run.py            # LAMMPS运行接口
│   │   └── run_iter.py              # 迭代流程控制
│   ├── dimer.py                     # 二聚体搜索相关功能
│   ├── __init__.py
│   ├── input.lammps                 # LAMMPS输入模板
│   ├── main.py                      # 主程序入口
│   └── model_devi.py                # 模型偏差计算
├── input/
│   ├── machine.json                 # 机器配置文件
│   ├── OC_10M.pb                    # 初始势文件
│   ├── par.json                     # 参数配置文件
│   ├── POSCAR                       # 初始结构文件(VASP格式)
│   └── ...
├── iter.xxxxxx/                     # 迭代目录,xxxxxx为迭代编号
│   └── 00.train/
│       ├── 000/
│       │   └── input.json
│       ├── 001/
│       │   └── input.json
│       ├── data.init -> path/to/init/data
│       └── data.iters
└── run.py                           # 运行脚本
```

- `auto/`: 主要源代码目录
    - `config/`: 配置管理
    - `fp_interface/`: 支持的势函数软件接口
    - `utils/`: 工具函数
    - `workflow/`: 迭代流程控制
    - `main.py`: 主程序入口
- `input/`: 输入文件目录
    - `par.json`: DPGenOpt参数配置
    - `machine.json`: 机器资源配置
    - `POSCAR`: 初始结构文件
- `iter.xxxxxx/`: 迭代目录
    - `00.train/`: 训练数据和配置
- `run.py`: 运行脚本
