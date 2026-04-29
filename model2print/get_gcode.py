import argparse
import subprocess
import sys
from pathlib import Path

import trimesh


A1_MINI_BUILD_VOLUME = (180.0, 180.0, 180.0)  # 单位：mm


def find_bambu_studio() -> Path:
    """
    在 Ubuntu 上自动寻找 Bambu Studio 可执行文件。
    支持：
    1. AppImage
    2. /usr/bin/bambu-studio
    3. /usr/local/bin/bambu-studio
    """

    candidates = [
        Path("/home/smmg/Bambu_Studio_linux.AppImage"),
        Path("/home/smmg/BambuStudio.AppImage"),
    ]

    for path in candidates:
        if path.exists():
            return path.resolve()

    # 额外搜索当前目录下所有 AppImage
    current_dir = Path.cwd()
    appimages = list(current_dir.glob("*.AppImage"))
    for app in appimages:
        if "Bambu" in app.name or "bambu" in app.name:
            return app.resolve()

    raise FileNotFoundError(
        "没有找到 Bambu Studio 可执行文件。\n"
        "请使用 --bambu 手动指定，例如：\n"
        "python auto_slice_bambu_ubuntu.py "
        "--bambu /home/yourname/bambu/Bambu_Studio_linux.AppImage"
    )


def ensure_executable(path: Path):
    """
    确保 AppImage 或可执行文件具有执行权限。
    """

    if path.suffix == ".AppImage":
        path.chmod(path.stat().st_mode | 0o111)


def load_stl_mesh(stl_path: Path) -> trimesh.Trimesh:
    """
    加载 STL 文件。
    如果 trimesh 读出来的是 Scene，则合并成一个 Trimesh。
    """

    obj = trimesh.load(stl_path)

    if isinstance(obj, trimesh.Scene):
        meshes = []

        for geometry in obj.geometry.values():
            if isinstance(geometry, trimesh.Trimesh):
                meshes.append(geometry)

        if not meshes:
            raise ValueError("STL 文件中没有有效 mesh。")

        mesh = trimesh.util.concatenate(meshes)

    elif isinstance(obj, trimesh.Trimesh):
        mesh = obj

    else:
        raise TypeError("无法识别 STL 文件内容。")

    return mesh


def inspect_stl(stl_path: Path, strict_watertight: bool = False):
    """
    检查 STL 基本信息：
    1. 尺寸是否超过 A1 mini 成型空间
    2. 是否封闭
    3. 法向是否一致
    """

    print("\n========== 1. STL 模型检查 ==========")

    mesh = load_stl_mesh(stl_path)

    bounds = mesh.bounds
    size = bounds[1] - bounds[0]

    print(f"STL 文件: {stl_path}")
    print(f"顶点数量: {len(mesh.vertices)}")
    print(f"三角面数量: {len(mesh.faces)}")
    print(f"是否封闭 watertight: {mesh.is_watertight}")
    print(f"法向是否一致 winding consistent: {mesh.is_winding_consistent}")
    print(f"模型最小坐标: {bounds[0]}")
    print(f"模型最大坐标: {bounds[1]}")
    print(f"模型尺寸 X/Y/Z: {size[0]:.3f}, {size[1]:.3f}, {size[2]:.3f} mm")

    build_x, build_y, build_z = A1_MINI_BUILD_VOLUME

    if size[0] > build_x or size[1] > build_y or size[2] > build_z:
        raise ValueError(
            "模型尺寸超过 A1 mini 打印范围。\n"
            f"模型尺寸: {size}\n"
            f"A1 mini 打印范围: {A1_MINI_BUILD_VOLUME}"
        )

    if not mesh.is_watertight:
        message = "警告：STL 不是完全封闭体，切片可能失败或产生异常路径。"
        if strict_watertight:
            raise ValueError(message)
        else:
            print(message)

    print("STL 检查完成。")


def run_cli_command(cmd, log_path: Path):
    """
    执行 Bambu Studio CLI 命令，并保存日志。
    """

    print("\n========== 2. 调用 Bambu Studio CLI ==========")
    print("执行命令:")

    printable_cmd = " ".join(
        f'"{str(x)}"' if " " in str(x) else str(x)
        for x in cmd
    )
    print(printable_cmd)

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=False,
    )

    log_path.parent.mkdir(parents=True, exist_ok=True)

    log_content = (
        "COMMAND:\n"
        + printable_cmd
        + "\n\nSTDOUT:\n"
        + result.stdout
        + "\n\nSTDERR:\n"
        + result.stderr
        + f"\n\nRETURN CODE: {result.returncode}\n"
    )

    log_path.write_text(log_content, encoding="utf-8")

    if result.stdout.strip():
        print("\nSTDOUT:")
        print(result.stdout)

    if result.stderr.strip():
        print("\nSTDERR:")
        print(result.stderr)

    print(f"\n日志文件已保存到: {log_path}")

    if result.returncode != 0:
        raise RuntimeError(
            f"Bambu Studio CLI 执行失败，返回码: {result.returncode}\n"
            f"请查看日志文件: {log_path}"
        )


def slice_stl_to_3mf(
    bambu_path: Path,
    stl_path: Path,
    machine_json: Path,
    process_json: Path,
    filament_json: Path,
    output_3mf: Path,
    scale: float = 1.0,
    use_xvfb: bool = False,
):
    """
    使用 Bambu Studio CLI 将 STL 切片为 3MF。
    """

    output_3mf.parent.mkdir(parents=True, exist_ok=True)

    bambu_cmd = [
        str(bambu_path),

        # 输出日志等级
        "--debug", "2",

        # 自动调整模型方向
        "--orient",

        # 自动摆放到打印板
        "--arrange", "1",

        # 模型缩放比例
        "--scale", str(scale),

        # 加载打印机配置和工艺配置
        "--load-settings", f"{machine_json};{process_json}",

        # 加载耗材配置
        "--load-filaments", str(filament_json),

        # 切片所有 plate
        "--slice", "0",

        # 导出 sliced 3MF
        "--export-3mf", str(output_3mf),

        # 输入 STL
        str(stl_path),
    ]

    if use_xvfb:
        cmd = ["xvfb-run", "-a"] + bambu_cmd
    else:
        cmd = bambu_cmd

    log_path = output_3mf.with_suffix(".slice.log.txt")

    run_cli_command(cmd, log_path)

    if not output_3mf.exists():
        raise FileNotFoundError(
            f"Bambu Studio CLI 没有报错，但没有找到输出文件: {output_3mf}"
        )

    print("\n========== 3. 切片完成 ==========")
    print(f"输入 STL: {stl_path}")
    print(f"输出 3MF: {output_3mf}")
    print(f"日志文件: {log_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ubuntu 下使用 Bambu Studio CLI 自动将 STL 切片为 3MF。"
    )

    parser.add_argument(
        "--stl",
        default="/home/smmg/AAM/construction/data/completion_result/repair_model.stl",
        help="输入 STL 文件路径，默认 repair_block_4mm.stl",
    )

    parser.add_argument(
        "--bambu",
        default=None,
        help="Bambu Studio 可执行文件路径，例如 /home/user/bambu/Bambu_Studio_linux.AppImage",
    )

    parser.add_argument(
        "--machine",
        default="/home/smmg/AAM/config/printing/printer_preset.json",
        help="打印机配置 JSON，默认 /home/smmg/AAM/config/printing/printer_preset.json",
    )

    parser.add_argument(
        "--process",
        default="/home/smmg/AAM/config/printing/process_preset.json",
        help="工艺配置 JSON，默认 /home/smmg/AAM/config/printing/process_preset.json",
    )

    parser.add_argument(
        "--filament",
        default="/home/smmg/AAM/config/printing/filament_preset.json",
        help="耗材配置 JSON，默认 /home/smmg/AAM/config/printing/filament_preset.json",
    )

    parser.add_argument(
        "--output",
        default="/home/smmg/AAM/model2print/data/slicer_output.3mf",
        help="输出 3MF 文件路径，默认 /home/smmg/AAM/model2print/data/slicer_output.3mf",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="模型缩放比例，默认 1.0",
    )

    parser.add_argument(
        "--strict-watertight",
        action="store_true",
        help="如果 STL 不是封闭体，则停止切片",
    )

    parser.add_argument(
        "--use-xvfb",
        action="store_true",
        help="在无图形界面的 Ubuntu 服务器上使用 xvfb-run 运行 Bambu Studio",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    stl_path = Path(args.stl).resolve()
    machine_json = Path(args.machine).resolve()
    process_json = Path(args.process).resolve()
    filament_json = Path(args.filament).resolve()
    output_3mf = Path(args.output).resolve()

    if args.bambu is None:
        bambu_path = find_bambu_studio()
    else:
        bambu_path = Path(args.bambu).resolve()

    required_files = {
        "STL 模型文件": stl_path,
        "Bambu Studio 可执行文件": bambu_path,
        "打印机配置 machine JSON": machine_json,
        "工艺配置 process JSON": process_json,
        "耗材配置 filament JSON": filament_json,
    }

    for name, path in required_files.items():
        if not path.exists():
            raise FileNotFoundError(f"{name} 不存在: {path}")

    ensure_executable(bambu_path)

    inspect_stl(
        stl_path=stl_path,
        strict_watertight=args.strict_watertight,
    )

    slice_stl_to_3mf(
        bambu_path=bambu_path,
        stl_path=stl_path,
        machine_json=machine_json,
        process_json=process_json,
        filament_json=filament_json,
        output_3mf=output_3mf,
        scale=args.scale,
        use_xvfb=args.use_xvfb,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\n========== 程序出错 ==========")
        print(error)
        sys.exit(1)