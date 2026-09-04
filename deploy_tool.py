import os
import time
import zipfile
import shutil
import tempfile

from model_artifacts import (
    create_deployment_package,
    discover_model_repository,
    get_model_iteration_mismatch,
    install_model_artifact_bundle,
    safe_extract_zip,
    validate_deployment_package,
    validate_deployment_package_filename,
    validate_package_name,
)

DEPLOY_DIR = "./deploy_packages"

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    clear_screen()
    print("==================================================")
    print("      📦 Galatea 模型部署与打包工具 V3.0.0      ")
    print("==================================================\n")

def pack_model():
    print_header()
    print(">>> [1] 打包新模型 (.gkg) <<<\n")
    
    os.makedirs("./models", exist_ok=True)
    os.makedirs(DEPLOY_DIR, exist_ok=True)
    
    repository = discover_model_repository("./models")
    pool_ids = sorted(repository["pools"])
    if not pool_ids:
        print("❌ 未在 ./models 目录下发现可验证的模型 UUID 池，按回车返回...")
        input()
        return

    print("发现以下模型 UUID 池：")
    for index, model_id in enumerate(pool_ids):
        pool = repository["pools"][model_id]
        print(
            f"  [{index + 1}] {', '.join(pool['prefixes'])} | {model_id} | "
            f"{len(pool['iterations'])} 个轮次"
        )
    choice = input("\n请先选择模型池序号: ")
    try:
        selected_pool_index = int(choice.strip()) - 1
        if selected_pool_index < 0 or selected_pool_index >= len(pool_ids):
            raise IndexError
        selected_pool_id = pool_ids[selected_pool_index]
    except (ValueError, IndexError):
        print("❌ 输入无效，按回车返回...")
        input()
        return

    pool_artifacts = repository["pools"][selected_pool_id]["artifacts"]
    print("\n该模型池包含以下可选主文件：")
    for index, artifact in enumerate(pool_artifacts):
        print(
            f"  [{index + 1}] iter {artifact['iteration']} | "
            f"{artifact['format']} | {artifact['primary']}"
        )
    choice = input("\n请选择文件序号 (多选请用逗号分隔，如 1,3): ")
    try:
        indices = [int(item.strip()) - 1 for item in choice.split(",")]
        if (
            not indices
            or len(indices) != len(set(indices))
            or any(index < 0 or index >= len(pool_artifacts) for index in indices)
        ):
            raise IndexError
        selected_artifacts = [pool_artifacts[index] for index in indices]
        selected_models = [artifact["primary"] for artifact in selected_artifacts]
    except (ValueError, IndexError):
        print("❌ 输入无效，按回车返回...")
        input()
        return
    mismatch = get_model_iteration_mismatch(selected_artifacts)
    if mismatch:
        print(f"❌ {mismatch}，请补齐相同轮次或只选择一种格式。")
        input("\n按回车键返回主菜单...")
        return

    default_name = os.path.splitext(selected_models[0])[0]
    pkg_name = input(f"\n请输入自定义包名 (直接回车默认使用 {default_name}): ").strip()
    if not pkg_name:
        pkg_name = default_name
        
    pkg_name += f"_{int(time.time())}" # 加时间戳防重名
    try:
        validate_package_name(pkg_name)
    except ValueError as error:
        print(f"❌ 包名不合法: {error}")
        input("\n按回车键返回主菜单...")
        return
    target_zip = os.path.join(DEPLOY_DIR, f"{pkg_name}.gkg")

    print("\n⏳ 正在极速封装打包中，请稍候...")

    try:
        extra_files = {}
        if os.path.exists("knowledge_base.json"):
            extra_files["knowledge_base.json"] = "knowledge_base.json"
            if os.path.exists("hash_mapping_report.json"):
                extra_files["hash_mapping_report.json"] = "hash_mapping_report.json"
        code_semantic_files = ("code_embeddings.npy", "code_embeddings_idx.json")
        if all(os.path.exists(filename) for filename in code_semantic_files):
            for filename in code_semantic_files:
                extra_files[filename] = filename
        elif any(os.path.exists(filename) for filename in code_semantic_files):
            raise FileNotFoundError("代码语义向量或索引缺失，拒绝打包不完整语义资产")
        if os.path.exists("meta_staples.json"):
            extra_files["meta_staples.json"] = "meta_staples.json"
        create_deployment_package(
            target_zip,
            "./models",
            selected_models,
            package_name=pkg_name,
            extra_files=extra_files,
        )
        print(f"\n✅ 打包大功告成！部署包已生成至: {os.path.abspath(target_zip)}")
    except Exception as e:
        print(f"\n❌ 打包过程中发生错误: {e}")
        
    input("\n按回车键返回主菜单...")

def unpack_model():
    print_header()
    print(">>> [2] 解包并导入系统 (.gkg) <<<\n")
    
    os.makedirs(DEPLOY_DIR, exist_ok=True)
    os.makedirs("./models", exist_ok=True)
    
    gkgs = []
    for filename in os.listdir(DEPLOY_DIR):
        try:
            validate_deployment_package_filename(filename)
            package_path = os.path.join(DEPLOY_DIR, filename)
            if os.path.isfile(package_path) and not os.path.islink(package_path):
                gkgs.append(filename)
        except ValueError:
            continue
    if not gkgs:
        print(f"❌ 未在 {DEPLOY_DIR} 下发现 .gkg 部署包。")
        print("你可以将包拖入该文件夹中再重试。按回车返回...")
        input()
        return
        
    print("发现以下部署包：")
    for i, g in enumerate(gkgs):
        size = os.path.getsize(os.path.join(DEPLOY_DIR, g)) / (1024*1024)
        print(f"  [{i+1}] {g} ({size:.1f} MB)")
        
    choice = input("\n请选择要解包导入的序号: ")
    try:
        idx = int(choice.strip()) - 1
        if idx < 0 or idx >= len(gkgs):
            raise IndexError
        selected_pkg = gkgs[idx]
    except (ValueError, IndexError):
        print("❌ 输入无效，按回车返回...")
        input()
        return
        
    pkg_path = os.path.join(DEPLOY_DIR, selected_pkg)
    
    print("\n⚠️ 警告：导入操作将直接覆盖当前系统的知识库文件 (knowledge_base.json等)。")
    confirm = input("确认继续导入吗？(y/n): ").strip().lower()
    
    if confirm != 'y':
        print("\n已取消导入。按回车返回...")
        input()
        return
        
    print("\n⏳ 正在原生极速解压部署中...")
    
    try:
        with tempfile.TemporaryDirectory(prefix="galatea_import_", dir=DEPLOY_DIR) as stage_dir:
            with zipfile.ZipFile(pkg_path, 'r') as gkg_zip:
                safe_extract_zip(gkg_zip, stage_dir)
            validated = validate_deployment_package(stage_dir)
            model_ids = sorted({record["model_id"] for record in validated["records"]})
            primary_models = validated["manifest"]["models_included"]
            if primary_models:
                installed = install_model_artifact_bundle(
                    stage_dir,
                    "./models",
                    primary_models,
                    expected_model_id=model_ids[0],
                )
                for filename in installed["files"]:
                    print(f"  -> 提取模型产物至 ./models/: {filename}")

            for filename in (
                "knowledge_base.json",
                "hash_mapping_report.json",
                "code_embeddings.npy",
                "code_embeddings_idx.json",
                "meta_staples.json",
            ):
                source = os.path.join(stage_dir, filename)
                if os.path.isfile(source):
                    print(f"  -> 覆盖系统基座文件: {filename}")
                    destination = os.path.abspath(filename)
                    with tempfile.NamedTemporaryFile(
                        prefix=f".{filename}.",
                        suffix=".import.tmp",
                        dir=os.path.dirname(destination),
                        delete=False,
                    ) as temporary:
                        temporary_path = temporary.name
                    try:
                        shutil.copy2(source, temporary_path)
                        os.replace(temporary_path, destination)
                    finally:
                        if os.path.exists(temporary_path):
                            os.remove(temporary_path)
                    
        print("\n✅ 系统环境更新完毕！模型和字典已全部就位。")
    except Exception as e:
        print(f"\n❌ 解压覆盖过程中发生错误: {e}")
        
    input("\n按回车键返回主菜单...")

def main_menu():
    while True:
        print_header()
        print("1. 📦 打包新模型 (Export .gkg)")
        print("2. 📥 解包并导入系统 (Import .gkg)")
        print("3. 🚪 退出 (Exit)\n")
        
        choice = input("请选择操作 [1/2/3]: ").strip()
        if choice == '1':
            pack_model()
        elif choice == '2':
            unpack_model()
        elif choice == '3':
            break
        else:
            print("❌ 无效选择，请重试...")
            time.sleep(1)

if __name__ == "__main__":
    main_menu()
