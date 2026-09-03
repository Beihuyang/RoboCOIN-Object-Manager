#!/usr/bin/env python3
"""Run and report the object + dataset/current-episode meta prompt experiment."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from project_paths import resolve_project_path
from semantic_prompts import semantic_noun_prompts
from stage1_track_select import DEFAULT_CKPT, detect_first_frame, load_sam3_detector
from reports.compare_semantic_prompts import _render
from reports.compare_single_prompt_nouns import _key, _load_result, _save_result

AB_ROOT = BASE_DIR / "reports/2026-09-03_single_prompt_nouns_ab_100"
BROAD_ROOT = BASE_DIR / "reports/2026-09-03_noun_prompt_three_way_100"
OUTPUT = BASE_DIR / "reports/2026-09-03_object_meta_prompt_100"
DATA_ROOT = BASE_DIR / "RoboCOIN_datasets"


def mean(items):
    return float(np.mean([float(x["score"]) for x in items])) if items else None


def metrics(groups):
    scores = [float(x["score"]) for group in groups for x in group]
    counts = [len(group) for group in groups]
    q = lambda x: round(float(np.quantile(scores, x)), 4) if scores else None
    return {"nonempty_sessions": sum(bool(x) for x in groups), "objects": sum(counts),
            "mean_objects": round(float(np.mean(counts)), 3),
            "median_objects": round(float(np.median(counts)), 3),
            "mean_confidence": round(float(np.mean(scores)), 4) if scores else None,
            "median_confidence": q(.5), "confidence_p25": q(.25), "confidence_p75": q(.75)}


def panel(frame, session, prompts, variants):
    colors = ["#76a7df", "#ffad5c", "#b28bd4", "#72d28e"]
    views = [_render(frame, [{**x, "prompt": label} for x in items], [x["mask"] for x in items])
             for label, items in variants]
    width, height, header = max(x.width for x in views), max(x.height for x in views), 116
    canvas = Image.new("RGB", (4 * width, height + header), (18, 22, 29)); draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), session, fill="white")
    for i, ((label, items), view) in enumerate(zip(variants, views)):
        canvas.paste(view, (i * width + (width-view.width)//2, header))
        confidence = "N/A" if mean(items) is None else f"{mean(items):.3f}"
        draw.text((i*width+12, 34), f"{label} | masks={len(items)} | mean={confidence}", fill=colors[i])
    text = ", ".join(prompts)
    draw.text((12, 64), "new prompts: " + (text[:300] + "..." if len(text)>300 else text), fill=(195,202,214))
    draw.text((12, 91), "Same 2K frame; threshold=0.2", fill=(195,202,214))
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--gpu", type=int, default=0); parser.add_argument("--threshold", type=float, default=.2)
    args = parser.parse_args(); output = args.output.resolve()
    source_rows = list(csv.DictReader((AB_ROOT/"datasets.csv").open()))
    pending = [r for r in source_rows if not (output/"raw"/_key(r["session"])/"result.json").is_file()]
    model = processor = None; failures=[]
    if pending: model, processor = load_sam3_detector(args.checkpoint, args.gpu)
    try:
        for i,row in enumerate(source_rows,1):
            session=row["session"]; target=output/"raw"/_key(session)
            if (target/"result.json").is_file(): print(f"[{i}/100] resume {session}",flush=True); continue
            try:
                meta,_=_load_result(AB_ROOT/"raw"/_key(session)/"after")
                frame=resolve_project_path(meta["frame"]); video=DATA_ROOT/f"{session}.mp4"
                prompts=semantic_noun_prompts(video,DATA_ROOT)
                detections,report=detect_first_frame(frame,model,processor,prompts,args.threshold,exclude_robot_arms=False)
                _save_result(target,detections,{"session":session,"frame":str(frame.resolve()),"prompts":prompts,
                    "strategy":"object + dataset name + episode-aligned meta nouns","filter_report":report})
                print(f"[{i}/100] {session}: {len(prompts)} prompts -> {len(detections)} masks",flush=True)
            except Exception as exc:
                failures.append({"session":session,"error":f"{type(exc).__name__}: {exc}"}); print(f"FAILED {session}: {exc}",flush=True)
                if isinstance(exc,torch.OutOfMemoryError): gc.collect();torch.cuda.empty_cache()
    finally:
        if model is not None: del model,processor;gc.collect();torch.cuda.empty_cache()

    loaded=[]; rows=[]
    for row in source_rows:
        session=row["session"]; new_dir=output/"raw"/_key(session)
        if not (new_dir/"result.json").is_file(): continue
        meta,new=_load_result(new_dir); _,legacy=_load_result(AB_ROOT/"raw"/_key(session)/"before")
        _,joined=_load_result(AB_ROOT/"raw"/_key(session)/"after")
        _,broad=_load_result(BROAD_ROOT/"raw"/_key(session)/"independent")
        frame=resolve_project_path(meta["frame"])
        record={"session":session,"prompt_count":len(meta["prompts"]),"legacy_count":len(legacy),
                "joined_count":len(joined),"broad_count":len(broad),"object_meta_count":len(new),
                "legacy_mean_confidence":mean(legacy),"joined_mean_confidence":mean(joined),
                "broad_mean_confidence":mean(broad),"object_meta_mean_confidence":mean(new)}
        rows.append(record);loaded.append((record,frame,meta["prompts"],legacy,joined,broad,new))
    names=["legacy","comma_joined","broad_independent","object_meta"]
    labels=["Legacy","Comma joined","All-text nouns","Object + meta"]
    groups={name:[x[i] for x in loaded] for name,i in zip(names,[3,4,5,6])}
    summary={"generated_at":datetime.now(timezone.utc).isoformat(),"completed_sessions":len(rows),
             "failed_sessions":len(failures),"threshold":args.threshold,
             "metrics":{name:metrics(group) for name,group in groups.items()},
             "average_new_prompt_count":round(float(np.mean([r["prompt_count"] for r in rows])),2)}
    calls=raw=duplicates=boundary=0
    for row in rows:
        data=json.loads((output/"raw"/_key(row["session"])/"result.json").read_text())["filter_report"]
        sem=data["semantic_discovery"];calls+=sem["sam3_detection_calls"];raw+=sum(sem["raw_counts"].values())
        duplicates+=sem["cross_prompt_duplicates_removed"];boundary+=data["boundary_filter"]["objects_removed"]
    summary["new_pipeline_processing"]={"text_prompt_calls":calls,"raw_candidates":raw,
        "cross_prompt_duplicates_removed":duplicates,"boundary_masks_removed":boundary}
    summary["object_meta_vs_legacy_sessions"] = {
        key: sum((int(r["object_meta_count"]) > int(r["legacy_count"])) if key == "gain" else
                 (int(r["object_meta_count"]) < int(r["legacy_count"])) if key == "loss" else
                 (int(r["object_meta_count"]) == int(r["legacy_count"])) for r in rows)
        for key in ("gain", "loss", "same")
    }
    output.mkdir(parents=True,exist_ok=True)
    (output/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False));(output/"failures.json").write_text(json.dumps(failures,indent=2,ensure_ascii=False))
    with (output/"datasets.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else []);writer.writeheader();writer.writerows(rows)

    colors=["#4c78a8","#f58518","#9467bd","#54a24b"]; m=summary["metrics"]
    fig,axes=plt.subplots(1,3,figsize=(16,4.8))
    for ax,key,title in zip(axes,["objects","nonempty_sessions","mean_confidence"],["Total masks","Sessions with masks","Mean confidence"]):
        bars=ax.bar(labels,[m[x][key] for x in names],color=colors);ax.bar_label(bars,fmt="%.3g",padding=3)
        ax.set_title(title);ax.grid(axis="y",alpha=.25);ax.tick_params(axis="x",rotation=18)
    fig.tight_layout();fig.savefig(output/"summary_metrics.png",dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    scores=[[float(x["score"]) for g in groups[name] for x in g] for name in names]
    axes[0].boxplot(scores,tick_labels=labels,showfliers=False);axes[0].set_ylim(0,1);axes[0].set_title("Confidence distribution");axes[0].grid(axis="y",alpha=.25);axes[0].tick_params(axis="x",rotation=18)
    x=np.arange(len(rows)); axes[1].scatter([r["legacy_count"] for r in rows],[r["object_meta_count"] for r in rows],alpha=.7,color=colors[-1])
    maxv=max([r["legacy_count"] for r in rows]+[r["object_meta_count"] for r in rows]);axes[1].plot([0,maxv],[0,maxv],"--",color="#d62728")
    axes[1].set_xlabel("Legacy masks");axes[1].set_ylabel("Object + meta masks");axes[1].set_title("Per-session mask counts");axes[1].grid(alpha=.25)
    fig.tight_layout();fig.savefig(output/"confidence_and_per_session.png",dpi=180);plt.close(fig)

    ranked=sorted(loaded,key=lambda x:abs(x[0]["object_meta_count"]-x[0]["legacy_count"]),reverse=True)
    target=output/"comparisons/highlights";target.mkdir(parents=True,exist_ok=True);cards=[]
    for rank,(row,frame,prompts,legacy,joined,broad,new) in enumerate(ranked[:24],1):
        name=f"{rank:02d}_{_key(row['session'])}.jpg"; panel(frame,row["session"],prompts,
            list(zip(labels,[legacy,joined,broad,new]))).save(target/name,quality=92)
        cards.append(f'<article><a href="comparisons/highlights/{name}"><imgless-than-sign>img src="comparisons/highlights/{name}"></a><b>{len(legacy)} → {len(new)}</b><span>{row["session"]}</span></article>'.replace('<less-than-sign>','<'))
    (output/"index.html").write_text('<!doctype html><meta charset="utf-8"><style>body{font:14px sans-serif;background:#10141b;color:#eef;padding:20px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(720px,1fr));gap:14px}article{background:#1b2330;padding:10px;border-radius:9px}img{width:100%}b,span{display:block;margin-top:6px}span{color:#9ba8ba}</style><h1>SAM3 prompts: 100-session comparison</h1><div class="grid">'+''.join(cards)+'</div>')
    old=m['legacy'];new=m['object_meta'];broad=m['broad_independent'];joined=m['comma_joined']
    (output/"REPORT.md").write_text(f'''# SAM3 `object + meta + 数据集名称` 100条实验

| 指标 | 旧版 | 名词逗号拼接 | meta+场景名词独立提示 | object+meta+数据集名称 |
|---|---:|---:|---:|---:|
| 有检出数据 | {old['nonempty_sessions']} | {joined['nonempty_sessions']} | {broad['nonempty_sessions']} | {new['nonempty_sessions']} |
| 掩码总数 | {old['objects']} | {joined['objects']} | {broad['objects']} | {new['objects']} |
| 每条平均掩码 | {old['mean_objects']} | {joined['mean_objects']} | {broad['mean_objects']} | {new['mean_objects']} |
| 平均置信度 | {old['mean_confidence']} | {joined['mean_confidence']} | {broad['mean_confidence']} | {new['mean_confidence']} |
| 中位置信度 | {old['median_confidence']} | {joined['median_confidence']} | {broad['median_confidence']} | {new['median_confidence']} |

新方案仅使用通用 `object`、数据集名称以及当前 episode 对应的 `meta/tasks.jsonl` / `meta/episodes.jsonl` 任务名词，不读取场景描述和 subtask。100条共调用 {calls} 次文本提示，产生 {raw} 个阈值内候选，跨提示去重 {duplicates} 个、边界过滤 {boundary} 个，最终保留 {new['objects']} 个。

相对旧版，掩码变化 {new['objects']-old['objects']:+d}（{(new['objects']-old['objects'])*100/max(1,old['objects']):+.1f}%），有检出数据变化 {new['nonempty_sessions']-old['nonempty_sessions']:+d} 条，平均置信度变化 {new['mean_confidence']-old['mean_confidence']:+.4f}。逐条看，掩码增加 {summary['object_meta_vs_legacy_sessions']['gain']} 条、减少 {summary['object_meta_vs_legacy_sessions']['loss']} 条、相同 {summary['object_meta_vs_legacy_sessions']['same']} 条。

相对“meta+场景描述全部名词独立提示”，新方案虽然平均提示数由8.94降到 {summary['average_new_prompt_count']}，但因为重新加入覆盖很广的 `object`，最终掩码仍增加 {new['objects']-broad['objects']} 个。因此不能根据总数断言误检减少；需要结合逐图审核。新方案的优势是提示来源明确、没有 `area/location/reference` 等场景描述噪声，并将有检出数据提升到 {new['nonempty_sessions']} 条。

![总体对比](summary_metrics.png)

![置信度与逐条数量](confidence_and_per_session.png)

所有方案使用同一批100条数据、相同2K帧和0.2阈值。旧版还包含当时的机械臂过滤等完整流程，并非严格单变量对照。没有真值掩码，因此总数、覆盖率和置信度只能比较输出行为，不能直接证明准确率；精选逐图对比见 `index.html`。
''')
    print(json.dumps(summary,indent=2,ensure_ascii=False))

if __name__=="__main__": main()
