# GKG部署包：共享内容与导入规范

> 适用于 Galatea-Core v3.16.7。本文是用户操作说明及供社区工具、Link读取器复用的公开容器合同，不改变模型协议V4、检查点3或网络结构。

## 选择哪一种包

| 导出内容 | 写出的容器 | 读取要求 |
|---|---|---|
| 原有模型、运行资产、可选语义源和142宣言追加池 | GKG4 | 原GKG4读取器或新版Core |
| 上述内容加卡组、重放、说明或预设中的任意文件 | GKG5 | 支持GKG5的新版Core或Link读取器 |

新版Core读取两种格式。没有共享附件时，继续写GKG4；不需要重新打包已有GKG4。
GKG5仅扩展容器，包内模型仍为V4，不要求重新训练或更换模型UUID。
默认包名`Galatea_V4_Pkg_<时间戳>.gkg`中的V4是模型协议，读取时应检查`manifest.json`。

模型主文件、ONNX外置权重、`card_vocab.json`及编译运行资产仍使用原有校验。
知识库/代码向量/索引是可选的接续语义源，三者不能拆开选择。GKG打包不自动重导出ONNX；
需要同步语义时，先使用[资源同步中的ONNX维护入口](code_semantics.md#onnx引擎更新)。

## 在WebUI导出与导入

1. 打开“模型部署与打包”，先选择同一UUID下的模型文件
2. 展开“可选共享内容”，在对应分类中勾选携带，读取本地来源目录；选择整目录或单个文件，重叠选择会去重
3. 说明文档请同时选择相对链接指向的页面和图片。预设可使用现有JSON、YAML、TOML等文件，本批不定义其参数结构
4. 确认附件数量和GKG4/5标识，再生成本地部署包
5. 在“解包与选择性导入”中解包到暂存区，分别选择模型、语义维护组件和共享附件；附件默认不选，可选一类全部文件或逐个选择

只选择附件时，不会安装模型、词表、运行语义或接续源。导入完成后，“最近导入的说明与预设”
可打开现有文档查看器：UTF-8 Markdown支持本地链接和图片；其他预设文本按原文显示，
不会自动应用训练、竞技场、联盟、温度或卡池参数。导入的Markdown不执行嵌入HTML。
PDF和其他较大附件可在本地目录查看；文本预览上限为4 MiB。

## 文件会存到哪里

每次导入按包名、模型引用和附件清单生成稳定的包分组。同一包重复导入时，相同内容复用；
已有文件内容不同则报告冲突，不覆盖。附件安装失败会回收本次新增文件，保留此前已导入的内容。
这是附件自身的处理规则，不表示模型、根资产和附件之间存在一个整包事务。

| 分类 | 包内相对路径示例 | Core导入位置与行为 |
|---|---|---|
| `decks` | `content/decks/烙印池/构筑.ydk` | `decks/GKG_<包分组>__<原目录标识>/构筑.ydk`；每个原目录建立独立物理池，同名卡组不会混到一起 |
| `replays` | `content/replays/holographic/Game1.json` | `ai_thoughts/GKG_<包分组>_<路径标识>_<原名>.json`，进入现有全息回放列表 |
| `replays` | `content/replays/raw/link_captures/duel.core.jsonl.gz` | `replays/imported/<包分组>/link_captures/duel.core.jsonl.gz`，保留原始字节 |
| `documents` | `content/documents/README.md` | `deploy_packages/imported/<包分组>/documents/README.md`，保留文档相对目录 |
| `presets` | `content/presets/deployment.yaml` | `deploy_packages/imported/<包分组>/presets/deployment.yaml`，仅保存和查看 |

卡组导入不覆盖`global_weights.json`或`virtual_pools.json`。新增物理池沿用现有默认权重规则，
可能参与后续随机抽样，可在资产与卡组管理中调整。YDK扩展名在导入时统一为小写以适配现有加载器，
原包内路径保留在导入清单中。卡组是否与本机CDB/词表兼容，继续由原预检判定；
附件携带不增加模型输入，也不单独赋予新卡学习能力。

原始录像的存储不等于新增格式解析器。能否转换和用于模仿学习，继续由现有录像入口按格式和完整性判定；
本次导入不会启动训练，也不会把原始轨迹自动送入PPO或BC。

导入清单自动写入`deploy_packages/imported/<包分组>/contents.json`，记录模型引用、原包内路径、
实际落点、大小和摘要。它无需手工维护；清理解包暂存区不会删除已导入文件。

## GKG5文件与清单合同

`.gkg`仍是ZIP。原GKG4模型和运行资产保持根目录布局，只有附件进入`content/`：

```text
manifest.json
galatea_iter_200.onnx
galatea_iter_200.onnx.data
card_vocab.json
semantic_lookup_v1.json
semantic_lookup_v1.npz
content/decks/烙印池/构筑.ydk
content/replays/holographic/Game1.json
content/replays/raw/duel.yrp3d
content/documents/README.md
content/documents/images/example.png
content/presets/deployment.yaml
```

完整清单沿用模型身份和协议字段，新增字段如下。示例摘要对应UTF-8文本
`#main\n100\n#extra\n!side\n`的23个字节：

```json
{
  "package_format_version": 5,
  "optional_content": {
    "format_version": 1,
    "files": [
      {
        "category": "decks",
        "path": "content/decks/Pool/A.ydk",
        "relative_path": "Pool/A.ydk",
        "size": 23,
        "sha256": "6969174311fe19974964da39551c0f1bdd8ff0d70ac639272f79cacda4fdc77d"
      }
    ]
  }
}
```

- `category`只能为`decks`、`replays`、`documents`、`presets`；`path`必须以对应的`content/<category>/`开头，`relative_path`必须等于余下路径
- `size`是未压缩字节数，`sha256`是实际入包字节的64位小写十六进制摘要；读取器应逐文件核对，不把校验和当作发布者签名
- GKG4不声明`optional_content`；GKG5要求附件清单格式1及非空文件列表。未声明的附件、缺失文件、摘要错误或未知附件格式不能导入
- 文件名支持中文、空格和相对目录；拒绝绝对路径、父目录、反斜杠、控制字符、跨平台保留名、符号链接、目录联接和大小写/Unicode等价路径冲突
- 模型和根资产仍使用原单层文件名规则。新包不允许把其他嵌套路径伪装成模型或协议资产
- 当前附件最多16,384个；GKG5总成员上限16,640，GKG4仍为256。展开总量64 GiB、单文件32 GiB；GKG5清单32 MiB，GKG4清单2 MiB，保留原压缩比限制

全息回放使用`replays`分类下的`holographic/`前缀且为JSON；原始文件推荐使用`raw/`前缀。
Core对其他`replays`相对路径按原始文件归档。说明允许Markdown、TXT、PDF和常见图片；
预设允许JSON、YAML、TOML、INI、CFG、TXT，不作为脚本执行。

## Core API与Link读取器

API位于[model_artifacts.py](../model_artifacts.py)，不新增独立运行模块：

```python
from model_artifacts import create_deployment_package, install_deployment_content

create_deployment_package(
    "deploy_packages/Shared.gkg", "models", ["galatea_iter_200.onnx"],
    package_name="Shared",
    optional_files={
        "decks": {"Pool/A.ydk": "decks/Pool/A.ydk"},
        "documents": {"README.md": "model_docs/README.md"},
        "presets": {"deployment.yaml": "presets/deployment.yaml"},
    },
)

# 在解包、核验后，只安装选中的附件
install_deployment_content(
    "deploy_packages/unpacked/Shared", ".",
    selected_paths=["content/documents/README.md"],
)
```

`discover_deployment_content`供文件选择使用；`safe_extract_zip`和`validate_deployment_package`
分别负责有界解包与完整核验；`plan_deployment_content_import`预览落点和已有文件状态。
`optional_files`由调用者显式提供“分类→相对路径→本地来源”，打包器不自动扫描并加入其他目录。

Link读取器可以按同一ZIP/JSON合同实现附件展示和选择导入，无需运行Core训练或语义生成流程。
它需识别GKG4/5、保留模型身份校验、验证附件清单和字节，并自行决定本地存储位置。
本批不为Link实现自动预设应用，也不改变Link真实服务器协议。
