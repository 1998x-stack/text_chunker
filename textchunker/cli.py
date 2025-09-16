from __future__ import annotations
import os
from loguru import logger
from .config import load_yaml, to_project_config, build_argparser, merge_cli
from .readers import load_inputs
from .factory import create_chunker
from .export import save_jsonl, save_txt_dir
from .visualization import show_chunks
from .chunkers import *  # noqa: F401  # 触发注册


def main() -> None:
    """命令行入口：读取 -> 分块 -> 可视化 -> 导出。"""
    parser = build_argparser()
    args = parser.parse_args()

    cfg = to_project_config(load_yaml(args.config))
    cfg = merge_cli(cfg, args)

    input_path = cfg.io.get("input_path")
    output_path = cfg.io.get("output_path")
    visualize = bool(cfg.project.get("visualize", False))
    export_format = cfg.project.get("export_format", "jsonl")

    logger.info(f"Strategy = {cfg.strategy.name}")
    chunker = create_chunker(cfg.strategy)

    for doc in load_inputs(input_path):
        logger.info(f"Processing: {doc.path}")
        chunks = chunker.chunk(doc.text)

        if visualize:
            show_chunks(chunks, title=os.path.basename(doc.path))

        if export_format in {"jsonl", "both"}:
            save_jsonl(output_path, chunks, source=doc.path)
            logger.info(f"Written JSONL to {output_path}")
        if export_format in {"txt_dir", "both"}:
            outdir = os.path.splitext(output_path)[0] + "_parts"
            save_txt_dir(outdir, chunks, prefix=os.path.basename(doc.path))
            logger.info(f"Written TXT parts to {outdir}")


if __name__ == "__main__":
    main()
