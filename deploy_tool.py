import os
import time
import zipfile
import shutil
import json
import tempfile

from model_artifacts import (
    assert_checkpoint_target_identity,
    build_package_model_records,
    collect_model_artifact_files,
    is_primary_model_filename,
    safe_extract_zip,
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
    
    models = sorted(f for f in os.listdir("./models") if is_primary_model_filename(f))
    if not models:
        print("❌ 未在 ./models 目录下发现 .pth 或 .onnx 模型文件，按回车返回...")
        input()
        return

    print("发现以下模型文件：")
    for i, m in enumerate(models):
        print(f"  [{i+1}] {m}")
        
    choice = input("\n请选择要打包的模型序号 (多选请用逗号分隔，如 1,3): ")
    try:
        indices = [int(x.strip()) - 1 for x in choice.split(',')]
        selected_models = [models[i] for i in indices]
    except:
        print("❌ 输入无效，按回车返回...")
        input()
        return

    try:
        package_model_files = collect_model_artifact_files("./models", selected_models)
        package_model_records = build_package_model_records("./models", selected_models)
    except Exception as error:
        print(f"❌ 模型产物不完整，已拒绝打包: {error}")
        input("\n按回车键返回主菜单...")
        return

    default_name = os.path.splitext(selected_models[0])[0]
    pkg_name = input(f"\n请输入自定义包名 (直接回车默认使用 {default_name}): ").strip()
    if not pkg_name:
        pkg_name = default_name
        
    pkg_name += f"_{int(time.time())}" # 加时间戳防重名
    target_zip = os.path.join(DEPLOY_DIR, f"{pkg_name}.gkg")

    print("\n⏳ 正在极速封装打包中，请稍候...")
    
    try:
        with zipfile.ZipFile(target_zip, 'w', zipfile.ZIP_DEFLATED) as gkg_zip:
            # 1. 压入模型
            for m in package_model_files:
                print(f"  -> 压缩模型: {m}")
                gkg_zip.write(os.path.join("./models", m), arcname=m)
                
            # 2. 压入字典 (如果存在)
            if os.path.exists("knowledge_base.json"):
                print("  -> 压缩知识库: knowledge_base.json")
                gkg_zip.write("knowledge_base.json", arcname="knowledge_base.json")
            if os.path.exists("meta_staples.json"):
                print("  -> 压缩兜底池: meta_staples.json")
                gkg_zip.write("meta_staples.json", arcname="meta_staples.json")
                
            # 3. 生成并压入清单
            manifest = {
                "package_name": pkg_name,
                "version": "3.0.0",
                "build_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "models_included": selected_models,
                "model_artifacts": package_model_records,
                "model_files_included": package_model_files,
            }
            print("  -> 生成清单: manifest.json")
            gkg_zip.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=4))
            
        print(f"\n✅ 打包大功告成！部署包已生成至: {os.path.abspath(target_zip)}")
    except Exception as e:
        print(f"\n❌ 打包过程中发生错误: {e}")
        
    input("\n按回车键返回主菜单...")

def unpack_model():
    print_header()
    print(">>> [2] 解包并导入系统 (.gkg) <<<\n")
    
    os.makedirs(DEPLOY_DIR, exist_ok=True)
    os.makedirs("./models", exist_ok=True)
    
    gkgs = [f for f in os.listdir(DEPLOY_DIR) if f.endswith(".gkg")]
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
        selected_pkg = gkgs[idx]
    except:
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

            staged_files = os.listdir(stage_dir)
            primary_models = [f for f in staged_files if is_primary_model_filename(f)]
            model_files = collect_model_artifact_files(stage_dir, primary_models)
            model_records = build_package_model_records(stage_dir, primary_models)
            for record in model_records:
                destination = os.path.join("./models", record["primary"])
                assert_checkpoint_target_identity(destination, record["model_id"])
            for filename in model_files:
                source = os.path.join(stage_dir, filename)
                destination = os.path.join("./models", filename)
                os.makedirs(os.path.dirname(destination), exist_ok=True)
                print(f"  -> 提取模型产物至 ./models/: {filename}")
                shutil.copy2(source, destination)

            for filename in staged_files:
                if filename.endswith(".json") and filename != "manifest.json" and not filename.endswith(".artifacts.json"):
                    print(f"  -> 覆盖系统基座文件: {filename}")
                    shutil.copy2(os.path.join(stage_dir, filename), filename)
                    
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
