import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import trimesh


A1_MINI_BUILD_VOLUME = (180.0, 180.0, 180.0)  # mm


def find_bambu_studio() -> Path:
    """自动寻找 Bambu Studio 可执行文件。"""
    candidates = [
        Path("/home/smmg/BambuStudio_ubuntu-22.04.AppImage"),
        Path("/home/smmg/BambuStudio.AppImage"),
        Path("/usr/bin/bambu-studio"),
        Path("/usr/local/bin/bambu-studio"),
    ]

    for path in candidates:
        if path.exists():
            return path.resolve()

    for app in Path.cwd().glob("*.AppImage"):
        if "bambu" in app.name.lower():
            return app.resolve()

    raise FileNotFoundError(
        "没有找到 Bambu Studio 可执行文件。请用 --bambu 手动指定。"
    )


def ensure_executable(path: Path):
    """确保 AppImage 有执行权限。"""
    if path.suffix == ".AppImage":
        path.chmod(path.stat().st_mode | 0o111)


def load_stl_mesh(stl_path: Path) -> trimesh.Trimesh:
    """加载 STL；如果读成 Scene，则合并全部 mesh。"""
    obj = trimesh.load(stl_path)

    if isinstance(obj, trimesh.Scene):
        meshes = [g for g in obj.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"STL 文件中没有有效 mesh: {stl_path}")
        return trimesh.util.concatenate(meshes)

    if isinstance(obj, trimesh.Trimesh):
        return obj

    raise TypeError(f"无法识别 STL 文件内容: {stl_path}")


def inspect_stl(stl_path: Path, strict_watertight: bool = False):
    """检查 STL 尺寸、封闭性、法向一致性。"""
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
        msg = "警告：STL 不是完全封闭体，切片可能失败或产生异常路径。"
        if strict_watertight:
            raise ValueError(msg)
        print(msg)

    print("STL 检查完成。")


def update_assemble_pos_z_from_orientation_meta(
    assemble_list_json: Path,
    orientation_meta: Path,
):
    """Set every object pos_z to full_box_z_height / 2 from orientation_meta.npz."""

    meta = np.load(orientation_meta, allow_pickle=True)
    full_box_z_height = float(np.asarray(meta["full_box_z_height"]).reshape(-1)[0])
    pos_z = full_box_z_height / 2.0 * 1000
    data = json.loads(assemble_list_json.read_text(encoding="utf-8"))
    updated_count = 0
    for plate in data.get("plates", []):
        for obj in plate.get("objects", []):
            obj["pos_z"] = [pos_z]
            obj["pos_x"] = [90]
            obj["pos_y"] = [90]
            updated_count += 1

    assemble_list_json.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n========== assemble_list pos_z update ==========")
    print("orientation_meta:", orientation_meta)
    print("full_box_z_height:", full_box_z_height)
    print("pos_z:", pos_z)
    print("updated objects:", updated_count)


def update_assemble_model_path(assemble_list_json: Path, input_model: Path):
    """Replace the first assemble-list object path with an external model path."""

    data = json.loads(assemble_list_json.read_text(encoding="utf-8"))
    updated = False
    for plate in data.get("plates", []):
        objects = plate.get("objects", [])
        if objects:
            objects[0]["path"] = str(input_model.resolve())
            updated = True
            break

    if not updated:
        raise ValueError(f"assemble_list.json has no object to update: {assemble_list_json}")

    assemble_list_json.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("\n========== assemble_list model path update ==========")
    print("input model:", input_model.resolve())
    print("assemble_list:", assemble_list_json)


def resolve_assemble_object_paths(assemble_list_json: Path) -> list[Path]:
    """
    从 assemble_list.json 中读取 STL 路径。
    相对路径按 assemble_list.json 所在目录解析，例如 path='print_model.stl'。
    """
    data = json.loads(assemble_list_json.read_text(encoding="utf-8"))
    base_dir = assemble_list_json.parent
    paths: list[Path] = []

    for plate in data.get("plates", []):
        plate_name = plate.get("plate_name", "unknown_plate")
        need_arrange = plate.get("need_arrange", None)
        if need_arrange not in [False, "false", "False", 0, "0", None]:
            print(f"警告：{plate_name} 的 need_arrange 不是 false，可能会触发自动摆盘。")

        for obj in plate.get("objects", []):
            raw_path = obj.get("path")
            if not raw_path:
                continue
            p = Path(raw_path)
            if not p.is_absolute():
                p = base_dir / p
            paths.append(p.resolve())

    if not paths:
        raise ValueError(f"assemble_list.json 中没有找到 object path: {assemble_list_json}")

    return paths


def inspect_assemble_list(assemble_list_json: Path, strict_watertight: bool = False):
    """检查 assemble_list.json 引用的 STL 是否存在，并检查模型。"""
    print("\n========== 0. assemble_list 检查 ==========")
    print(f"assemble_list: {assemble_list_json}")

    object_paths = resolve_assemble_object_paths(assemble_list_json)
    for p in object_paths:
        if not p.exists():
            raise FileNotFoundError(
                f"assemble_list.json 引用的 STL 不存在: {p}\n"
                "如果 path 是相对路径，请把 STL 放到 assemble_list.json 同一目录下。"
            )
        inspect_stl(p, strict_watertight=strict_watertight)


def format_cmd(cmd) -> str:
    return " ".join(f'"{str(x)}"' if " " in str(x) else str(x) for x in cmd)


def run_cli_command(cmd, log_path: Path, cwd: Path | None = None) -> int:
    """执行 Bambu Studio CLI，并保存 stdout/stderr 日志。"""
    print("\n========== 2. 调用 Bambu Studio CLI ==========")
    print("执行命令:")
    printable_cmd = format_cmd(cmd)
    print(printable_cmd)

    if cwd is not None:
        print(f"工作目录 cwd: {cwd}")

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=False,
        cwd=str(cwd) if cwd is not None else None,
    )

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_content = (
        "COMMAND:\n"
        + printable_cmd
        + f"\n\nCWD:\n{cwd}\n"
        + "\nSTDOUT:\n"
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
    return result.returncode


def validate_sliced_3mf(output_3mf: Path):
    """检查导出的 3MF 内是否包含已切片 gcode。"""
    if not output_3mf.exists():
        raise FileNotFoundError(f"没有找到输出 3MF: {output_3mf}")

    with zipfile.ZipFile(output_3mf, "r") as zf:
        names = zf.namelist()

    gcode_files = [
        n for n in names
        if n.startswith("Metadata/plate_") and n.endswith(".gcode")
    ]

    print("\n========== 3. 输出 3MF 检查 ==========")
    print(f"输出 3MF: {output_3mf}")
    print(f"内部 plate gcode: {gcode_files}")

    if not gcode_files:
        raise ValueError(
            "输出 3MF 中没有找到 Metadata/plate_*.gcode。\n"
            "这说明它可能不是 sliced 3MF，打印机可能不接受直接打印。"
        )


def slice_stl_to_3mf_with_assemble_list(
    bambu_path: Path,
    machine_json: Path,
    process_json: Path,
    filament_json: Path,
    assemble_list_json: Path,
    output_3mf: Path,
    scale: float = 1000.0,
    use_xvfb: bool = True,
):
    """
    使用 Bambu Studio CLI + assemble_list.json 切片。

    关键点：
    1. 不写 --orient，等价于 orient=0。
    2. 使用 --arrange 0，禁止自动摆盘。
    3. 使用 --load-assemble-list 加载模型与 plate 位姿。
    4. 命令末尾不再附加 STL 路径，因为 STL 由 assemble_list.json 指定。
    """
    output_3mf.parent.mkdir(parents=True, exist_ok=True)

    bambu_cmd = [
        str(bambu_path),
        "--debug", "2",

        # orient=0：官方 --orient 是 flag，不写即可关闭自动定向。

        # arrange=0：关闭自动摆盘，使用 assemble_list.json 中的 pos_x/pos_y/pos_z。
        "--arrange", "0",

        "--scale", str(scale),
        "--load-settings", f"{machine_json};{process_json}",
        "--load-filaments", str(filament_json),
        "--load-assemble-list", str(assemble_list_json),
        "--no-check",
        "--slice", "0",
        "--export-3mf", str(output_3mf),
    ]

    if use_xvfb:
        cmd = [
            "xvfb-run",
            "-a",
            "-s",
            "-screen 0 1920x1080x24 +extension GLX +render -noreset",
        ] + bambu_cmd
    else:
        cmd = bambu_cmd

    log_path = output_3mf.with_suffix(".slice.log.txt")

    # 重要：如果 assemble_list.json 里 path='print_model.stl' 是相对路径，
    # 就让 Bambu Studio 在 assemble_list 所在目录运行。
    return_code = run_cli_command(
        cmd,
        log_path=log_path,
        cwd=assemble_list_json.parent,
    )

    if not output_3mf.exists():
        raise RuntimeError(
            f"Bambu Studio CLI 未生成输出文件，返回码: {return_code}\n"
            f"请查看日志: {log_path}"
        )

    if return_code != 0:
        print(
            f"警告：Bambu Studio CLI 返回码不是 0: {return_code}，"
            "但输出文件已经生成，继续检查 3MF 内容。"
        )

    validate_sliced_3mf(output_3mf)

    print("\n========== 4. 切片完成 ==========")
    print(f"assemble_list: {assemble_list_json}")
    print(f"输出 3MF: {output_3mf}")
    print(f"日志文件: {log_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="使用 Bambu Studio CLI + assemble_list.json 自动切片为 sliced 3MF。"
    )

    parser.add_argument(
        "--bambu",
        default='/home/smmg/BambuStudio_ubuntu-22.04.AppImage',
        help="Bambu Studio 可执行文件路径，例如 /home/smmg/BambuStudio_ubuntu-22.04.AppImage",
    )

    parser.add_argument(
        "--machine",
        default="/home/smmg/AAM/config/printing/machine_FULL.json",
        help="打印机 full config JSON",
    )

    parser.add_argument(
        "--process",
        default="/home/smmg/AAM/config/printing/process_FULL.json",
        help="工艺 full config JSON",
    )

    parser.add_argument(
        "--filament",
        default="/home/smmg/AAM/config/printing/filament_FULL.json",
        help="耗材 full config JSON",
    )

    parser.add_argument(
        "--assemble-list",
        default="/home/smmg/AAM/config/printing/assemble_list.json",
        help="assemble_list.json 路径。里面应引用 print_model.stl，并设置 pos_x=0,pos_y=0,pos_z=0。",
    )

    parser.add_argument(
        "--input-model",
        default=None,
        help="External STL/model path used to update the first object in assemble_list.json.",
    )

    parser.add_argument(
        "--orientation-meta",
        default=None,
        help="completion/depression/orientation_meta.npz. If set, objects.pos_z is updated to full_box_z_height / 2 before slicing.",
    )

    parser.add_argument(
        "--output",
        default="/home/smmg/AAM/model2print/data/print_sliced.3mf",
        help="输出 sliced 3MF 文件路径",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=1000.0,
        help="模型缩放比例，默认 1.0。如果 STL 单位是米，可设为 1000。",
    )

    parser.add_argument(
        "--strict-watertight",
        action="store_true",
        help="如果 STL 不是封闭体，则停止切片。",
    )

    parser.add_argument(
        "--use-xvfb",
        dest="use_xvfb",
        action="store_true",
        help="使用 xvfb-run 运行 Bambu Studio。默认开启。",
    )

    parser.add_argument(
        "--no-xvfb",
        dest="use_xvfb",
        action="store_false",
        help="不使用 xvfb-run。",
    )

    parser.set_defaults(use_xvfb=True)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.bambu is None:
        bambu_path = find_bambu_studio()
    else:
        bambu_path = Path(args.bambu).resolve()

    machine_json = Path(args.machine).resolve()
    process_json = Path(args.process).resolve()
    filament_json = Path(args.filament).resolve()
    assemble_list_json = Path(args.assemble_list).resolve()
    input_model = Path(args.input_model).resolve() if args.input_model else None
    orientation_meta = Path(args.orientation_meta).resolve() if args.orientation_meta else None
    output_3mf = Path(args.output).resolve()

    required_files = {
        "Bambu Studio 可执行文件": bambu_path,
        "打印机配置 machine JSON": machine_json,
        "工艺配置 process JSON": process_json,
        "耗材配置 filament JSON": filament_json,
        "assemble_list JSON": assemble_list_json,
    }

    if orientation_meta is not None:
        required_files["orientation_meta NPZ"] = orientation_meta
    if input_model is not None:
        required_files["input model"] = input_model

    for name, path in required_files.items():
        if not path.exists():
            raise FileNotFoundError(f"{name} 不存在: {path}")

    ensure_executable(bambu_path)

    if input_model is not None:
        update_assemble_model_path(
            assemble_list_json=assemble_list_json,
            input_model=input_model,
        )

    if orientation_meta is not None:
        update_assemble_pos_z_from_orientation_meta(
            assemble_list_json=assemble_list_json,
            orientation_meta=orientation_meta,
        )

    inspect_assemble_list(
        assemble_list_json=assemble_list_json,
        strict_watertight=args.strict_watertight,
    )

    slice_stl_to_3mf_with_assemble_list(
        bambu_path=bambu_path,
        machine_json=machine_json,
        process_json=process_json,
        filament_json=filament_json,
        assemble_list_json=assemble_list_json,
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
