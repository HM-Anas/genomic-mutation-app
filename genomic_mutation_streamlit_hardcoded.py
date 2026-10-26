"""Streamlit dashboard for genomic mutation annotation and interval validation.

Coordinates are BED-style 0-based, half-open intervals. This app preserves
RegulomeDB, common-RBP, TSS overlap annotation, manual lookup, Fisher tests,
chromosome-based permutation tests, and CSV exports from the desktop tool.
"""
from __future__ import annotations
import bisect
import csv
import io
import random
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from scipy.stats import fisher_exact

APP_TITLE = "Genomic Mutation Annotation Tool"
DEFAULT_PERMUTATIONS = 10000

# Put all seven input files inside the project's ./data/ folder.
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
INPUT_FILES = {
    "mutations": DATA_DIR / "final_clean_intersection.bed",
    "regulomedb": DATA_DIR / "regulome_variants.bed",
    "rbp": DATA_DIR / "240_Common_RBP_Sites.bed",
    "tss": DATA_DIR / "TSS_plus_minus_10kb_sorted.bed",
    "reg_controls": DATA_DIR / "random_COSMIC_46_plus_minus_10bp.bed",
    "rbp_controls": DATA_DIR / "random_COSMIC_240_plus_minus_10bp.bed",
    "sizes": DATA_DIR / "hg38.chrom.sizes",
}


def norm_chr(chrom: str) -> str:
    chrom = str(chrom).strip()
    if not chrom:
        return ""
    return chrom if chrom.lower().startswith("chr") else "chr" + chrom


def parse_interval_file(uploaded_file) -> Tuple[List[dict], List[str]]:
    """Read a local file path or Streamlit upload; preserve extra columns."""
    if uploaded_file is None:
        return [], ["No file supplied."]
    if isinstance(uploaded_file, (str, __import__("pathlib").Path)):
        from pathlib import Path
        path = Path(uploaded_file)
        if not path.is_file():
            return [], [f"File not found: {path}"]
        name = path.name.lower()
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    else:
        name = uploaded_file.name.lower()
        raw = uploaded_file.getvalue()
        text = raw.decode("utf-8-sig", errors="replace")
    rows, issues = [], []
    reader = io.StringIO(text)
    for line_no, line in enumerate(reader, 1):
        line = line.strip()
        if not line or line.startswith(("#", "track", "browser")):
            continue
        if name.endswith(".csv"):
            fields = next(csv.reader([line]))
        elif "\t" in line:
            fields = line.split("\t")
        else:
            fields = line.split()
        if len(fields) < 3:
            issues.append(f"Line {line_no}: fewer than 3 columns; skipped")
            continue
        try:
            chrom, start, end = norm_chr(fields[0]), int(fields[1]), int(fields[2])
        except (ValueError, IndexError):
            # Headers are expected in some TSV/CSV files; report but don't fail.
            if line_no <= 3:
                issues.append(f"Line {line_no}: header or invalid coordinates; skipped")
            continue
        if not chrom or start < 0 or end <= start:
            issues.append(f"Line {line_no}: invalid interval {fields[:3]}; skipped")
            continue
        rows.append({"chrom": chrom, "start": start, "end": end, "fields": fields,
                     "source_line": line_no})
    if not rows:
        issues.append("No valid intervals were parsed from this file.")
    return rows, issues


class IntervalIndex:
    """Chromosome interval index using binary search and prefix maximum ends."""
    def __init__(self, rows: List[dict]):
        self.data, self.starts, self.prefix_max_end = {}, {}, {}
        for row in rows:
            self.data.setdefault(row["chrom"], []).append(row)
        for chrom, intervals in self.data.items():
            intervals.sort(key=lambda r: r["start"])
            self.starts[chrom] = [r["start"] for r in intervals]
            maxima, maximum = [], -1
            for row in intervals:
                maximum = max(maximum, row["end"])
                maxima.append(maximum)
            self.prefix_max_end[chrom] = maxima

    def find(self, chrom: str, start: int, end: int) -> List[dict]:
        chrom = norm_chr(chrom)
        intervals = self.data.get(chrom, [])
        if not intervals or end <= start:
            return []
        i = bisect.bisect_left(self.starts[chrom], end) - 1
        hits, prefix = [], self.prefix_max_end[chrom]
        while i >= 0 and prefix[i] > start:
            row = intervals[i]
            if row["end"] > start:
                hits.append(row)
            i -= 1
        hits.reverse()
        return hits


def overlap_details(index: IntervalIndex, chrom: str, start: int, end: int):
    hits = index.find(chrom, start, end)
    if not hits:
        return "NO", "", 0
    details = ["|".join(r["fields"][3:]) or f"{r['start']}-{r['end']}" for r in hits[:10]]
    if len(hits) > 10:
        details.append(f"... {len(hits)-10} more")
    return "YES", "; ".join(details), len(hits)


def load_mutations(rows: List[dict]) -> List[dict]:
    mutations = []
    for row in rows:
        f = row["fields"]
        mutations.append({
            "Chromosome": row["chrom"], "Start_0based": row["start"],
            "End_0based": row["end"], "Position_1based": row["start"] + 1,
            "Gene": f[3] if len(f) > 3 else "",
            "Cancer_type": f[4] if len(f) > 4 else "",
            "Count": f[5] if len(f) > 5 else "", "Candidate_match": "YES",
            "Manual_lookup": "NO",
        })
    return mutations


def annotate(mutations, reg_index, rbp_index, tss_index):
    results = []
    for i, mutation in enumerate(mutations, 1):
        chrom, start, end = mutation["Chromosome"], mutation["Start_0based"], mutation["End_0based"]
        reg, reg_details, reg_n = overlap_details(reg_index, chrom, start, end)
        rbp, rbp_details, rbp_n = overlap_details(rbp_index, chrom, start, end)
        tss, tss_details, tss_n = overlap_details(tss_index, chrom, start, end)
        if reg == "YES" and rbp == "YES": category = "Both regulatory + RBP"
        elif reg == "YES": category = "Regulatory only"
        elif rbp == "YES": category = "RBP only"
        else: category = "Neither"
        results.append({
            "Record_ID": f"VAR_{i:04d}", **mutation,
            "TSS_overlap": tss, "TSS_details": tss_details, "TSS_hit_count": tss_n,
            "RegulomeDB_overlap": reg, "RegulomeDB_details": reg_details, "RegulomeDB_hit_count": reg_n,
            "Common_RBP_overlap": rbp, "Common_RBP_details": rbp_details, "Common_RBP_hit_count": rbp_n,
            "Overlap_category": category,
        })
    return results


def deduplicate_records(mutations):
    """Deduplicate same coordinate/gene/cancer record, preserving distinct labels/counts."""
    unique, seen = [], set()
    for m in mutations:
        key = (m["Chromosome"], m["Start_0based"], m["End_0based"], m.get("Gene", ""),
               m.get("Cancer_type", ""), str(m.get("Count", "")))
        if key not in seen:
            seen.add(key); unique.append(m)
        elif m.get("Manual_lookup") == "YES":
            for existing in unique:
                if (existing["Chromosome"], existing["Start_0based"], existing["End_0based"],
                    existing.get("Gene", ""), existing.get("Cancer_type", ""),
                    str(existing.get("Count", ""))) == key:
                    existing["Manual_lookup"] = "YES"
                    break
    return unique


def read_chromosome_lengths(uploaded_file):
    lengths = {}
    if uploaded_file is None:
        return lengths
    if isinstance(uploaded_file, (str, __import__("pathlib").Path)):
        from pathlib import Path
        path = Path(uploaded_file)
        if not path.is_file():
            return lengths
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    else:
        text = uploaded_file.getvalue().decode("utf-8-sig", errors="replace")
    for line in text.splitlines():
        fields = line.strip().split()
        if len(fields) < 2: continue
        try:
            n = int(fields[1])
            if n > 0: lengths[norm_chr(fields[0])] = n
        except ValueError:
            continue
    return lengths


def validate_records(mutations, indexes, lengths=None):
    """Return row-level QC issues; interval coordinates must be 0-based half-open."""
    issues = []
    for i, m in enumerate(mutations, 1):
        chrom, start, end = m["Chromosome"], m["Start_0based"], m["End_0based"]
        if start < 0 or end <= start:
            issues.append({"Record": i, "Chromosome": chrom, "Start": start, "End": end,
                           "Issue": "Invalid BED interval"})
        if lengths is not None:
            if chrom not in lengths:
                issues.append({"Record": i, "Chromosome": chrom, "Start": start, "End": end,
                               "Issue": "Chromosome absent from chromosome-length file"})
            elif start >= lengths[chrom] or end > lengths[chrom]:
                issues.append({"Record": i, "Chromosome": chrom, "Start": start, "End": end,
                               "Issue": f"Interval exceeds chromosome length ({lengths[chrom]})"})
        for label, idx in indexes.items():
            if chrom not in idx.data:
                # A missing chromosome in a reference is informative, but not necessarily an error.
                pass
    return issues


def fisher_test(observed, controls, feature):
    if not observed or not controls:
        raise ValueError("Fisher's exact test needs non-empty observed and control datasets.")
    a = sum(r[feature] == "YES" for r in observed); b = len(observed) - a
    c = sum(r[feature] == "YES" for r in controls); d = len(controls) - c
    table = [[a, b], [c, d]]
    odds, p_two = fisher_exact(table, alternative="two-sided")
    _, p_one = fisher_exact(table, alternative="greater")
    return {"Feature": feature, "Observed_n": len(observed), "Observed_overlap": a,
            "Observed_rate_percent": 100*a/len(observed), "Control_n": len(controls),
            "Control_overlap": c, "Control_rate_percent": 100*c/len(controls),
            "Odds_ratio": odds, "Fisher_p_value": p_two, "Two_sided_p_value": p_two,
            "One_sided_p_value": p_one, "Contingency_table": str(table)}


def permutation_test(mutations, feature_index, chromosome_lengths, n_permutations=DEFAULT_PERMUTATIONS, seed=42):
    usable = [m for m in mutations if m["Chromosome"] in chromosome_lengths
              and 0 <= m["Start_0based"] < m["End_0based"] <= chromosome_lengths[m["Chromosome"]]]
    if not usable:
        raise ValueError("No valid candidate intervals match the chromosome-length file.")
    rng = random.Random(seed)
    observed = sum(bool(feature_index.find(m["Chromosome"], m["Start_0based"], m["End_0based"])) for m in usable)
    exceedances = 0
    for _ in range(n_permutations):
        random_count = 0
        for m in usable:
            chrom = m["Chromosome"]; length = m["End_0based"] - m["Start_0based"]
            max_start = chromosome_lengths[chrom] - length
            start = rng.randint(0, max_start); end = start + length
            if feature_index.find(chrom, start, end): random_count += 1
        if random_count >= observed: exceedances += 1
    return {"Observed_overlap": observed, "Tested_n": len(usable),
            "Observed_rate_percent": 100*observed/len(usable), "Permutations": n_permutations,
            "Permutation_p_value": (exceedances+1)/(n_permutations+1), "Random_seed": seed,
            "Null_model": "Uniform random positions within chromosomes"}


def exact_coordinate_matches(mutations, chrom, start, end):
    return [m for m in mutations if m["Chromosome"] == norm_chr(chrom)
            and m["Start_0based"] == int(start) and m["End_0based"] == int(end)]


def records_to_df(records):
    return pd.DataFrame(records) if records else pd.DataFrame()


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="🧬")
    st.title("🧬 Genomic Mutation Annotation Tool")
    st.caption("Coordinate lookup · interval validation · RegulomeDB/RBP/TSS annotation · statistical testing")
    st.info("Coordinates are interpreted as BED-style 0-based, half-open intervals: [start, end). Confirm all inputs use the same genome build (e.g., hg38).")

    with st.sidebar:
        st.header("Bundled input files")
        st.caption("Inputs load automatically from the project's data/ directory.")
        st.code("\\n".join(f"{key}: {path.name}" for key, path in INPUT_FILES.items()))
        st.divider()
        st.subheader("Validation settings")
        genome_build = st.selectbox("Declared genome build", ["hg38 / GRCh38", "hg19 / GRCh37", "Other / unknown"], index=0)
        n_permutations = st.number_input("Permutation count", min_value=100, max_value=100000, value=DEFAULT_PERMUTATIONS, step=100)
        seed = st.number_input("Random seed", min_value=0, max_value=2147483647, value=42, step=1)

    missing_files = [str(path) for path in INPUT_FILES.values() if not path.is_file()]
    if missing_files:
        st.error("Required bundled input files are missing. Add them to the data/ folder:")
        for missing in missing_files:
            st.code(missing)
        st.stop()

    mutations_file = INPUT_FILES["mutations"]
    reg_file = INPUT_FILES["regulomedb"]
    rbp_file = INPUT_FILES["rbp"]
    tss_file = INPUT_FILES["tss"]
    reg_ctrl_file = INPUT_FILES["reg_controls"]
    rbp_ctrl_file = INPUT_FILES["rbp_controls"]
    sizes_file = INPUT_FILES["sizes"]

    with st.spinner("Reading files and building interval indexes…"):
        mut_rows, mut_issues = parse_interval_file(mutations_file)
        reg_rows, reg_issues = parse_interval_file(reg_file)
        rbp_rows, rbp_issues = parse_interval_file(rbp_file)
        tss_rows, tss_issues = parse_interval_file(tss_file)
        reg_ctrl_rows, reg_ctrl_issues = parse_interval_file(reg_ctrl_file) if reg_ctrl_file else ([], [])
        rbp_ctrl_rows, rbp_ctrl_issues = parse_interval_file(rbp_ctrl_file) if rbp_ctrl_file else ([], [])
        lengths = read_chromosome_lengths(sizes_file) if sizes_file else None
        reg_index, rbp_index, tss_index = IntervalIndex(reg_rows), IntervalIndex(rbp_rows), IntervalIndex(tss_rows)
        mutations = deduplicate_records(load_mutations(mut_rows))
        results = annotate(mutations, reg_index, rbp_index, tss_index)

    parse_issues = []
    for label, issues in [("Mutation file",mut_issues),("RegulomeDB",reg_issues),("Common RBP",rbp_issues),("TSS",tss_issues)]:
        parse_issues.extend([{"File":label,"Issue":x} for x in issues])
    qc_issues = validate_records(mutations, {"RegulomeDB":reg_index,"Common RBP":rbp_index,"TSS":tss_index}, lengths)
    unique_positions = {(m["Chromosome"],m["Start_0based"],m["End_0based"]) for m in mutations}
    unique_genes = {m["Gene"] for m in mutations if m.get("Gene")}
    cancers = {m["Cancer_type"] for m in mutations if m.get("Cancer_type")}
    reg_n = sum(r["RegulomeDB_overlap"] == "YES" for r in results)
    rbp_n = sum(r["Common_RBP_overlap"] == "YES" for r in results)
    tss_n = sum(r["TSS_overlap"] == "YES" for r in results)
    st.success(f"Annotated {len(results):,} records · {len(unique_positions):,} unique genomic intervals")
    m1,m2,m3,m4,m5 = st.columns(5)
    m1.metric("Records", f"{len(results):,}")
    m2.metric("Unique intervals", f"{len(unique_positions):,}")
    m3.metric("Unique genes", f"{len(unique_genes):,}")
    m4.metric("RegulomeDB overlaps", f"{reg_n:,}", f"{reg_n/max(len(results),1):.1%}")
    m5.metric("Common-RBP overlaps", f"{rbp_n:,}", f"{rbp_n/max(len(results),1):.1%}")
    st.caption(f"TSS overlaps: {tss_n:,} ({tss_n/max(len(results),1):.1%}) · Cancer-type labels: {len(cancers):,} · Declared build: {genome_build}")

    tabs = st.tabs(["Coordinate lookup", "Annotated variants", "Validation & QC", "Statistics", "Exports"])
    with tabs[0]:
        st.subheader("Advanced coordinate lookup")
        c1,c2,c3 = st.columns([1,1,1])
        chrom = c1.text_input("Chromosome", value="chr6", help="Use chr6 or 6; names are normalized.")
        start = c2.number_input("Start (0-based)", min_value=0, value=6468364, step=1)
        end = c3.number_input("End (exclusive)", min_value=1, value=6468365, step=1)
        if st.button("Inspect coordinate", type="primary"):
            st.session_state["lookup_query"] = (norm_chr(chrom), int(start), int(end))
        query = st.session_state.get("lookup_query", ("chr6",6468364,6468365))
        qchrom,qstart,qend = query
        if qend <= qstart:
            st.error("Invalid interval: end must be greater than start.")
        else:
            exact = exact_coordinate_matches(mutations, qchrom, qstart, qend)
            reg_hits, rbp_hits, tss_hits = reg_index.find(qchrom,qstart,qend), rbp_index.find(qchrom,qstart,qend), tss_index.find(qchrom,qstart,qend)
            if lengths is not None:
                if qchrom not in lengths: st.warning(f"{qchrom} is absent from the uploaded chromosome-length file.")
                elif qstart >= lengths[qchrom] or qend > lengths[qchrom]: st.error(f"Interval exceeds {qchrom} length ({lengths[qchrom]:,}).")
                else: st.success(f"Coordinate is within {qchrom} length ({lengths[qchrom]:,}).")
            l1,l2,l3,l4 = st.columns(4)
            l1.metric("Exact master-record matches",len(exact))
            l2.metric("RegulomeDB interval hits",len(reg_hits))
            l3.metric("RBP interval hits",len(rbp_hits))
            l4.metric("TSS interval hits",len(tss_hits))
            st.caption(f"BED interval queried: {qchrom}:{qstart}-{qend} · 1-based position for a single base: {qstart+1}")
            if not exact: st.info("No exact interval match in the mutation file. This does not prevent annotation against reference intervals.")
            for label,hits in [("Exact candidate records",exact),("RegulomeDB overlapping intervals",reg_hits),("Common-RBP overlapping intervals",rbp_hits),("TSS overlapping intervals",tss_hits)]:
                with st.expander(f"{label} ({len(hits)})", expanded=(label=="Exact candidate records" or (label.startswith("RegulomeDB") and bool(hits)))):
                    if label == "Exact candidate records":
                        st.dataframe(pd.DataFrame(hits), use_container_width=True, hide_index=True) if hits else st.write("No exact record found.")
                    else:
                        show = [{"Chromosome":h["chrom"],"Start_0based":h["start"],"End_0based":h["end"],"Overlap_bp":max(0,min(qend,h["end"])-max(qstart,h["start"])),"Extra_columns":" | ".join(h["fields"][3:])} for h in hits]
                        st.dataframe(pd.DataFrame(show), use_container_width=True, hide_index=True) if show else st.write("No overlapping reference intervals.")
            st.markdown("**Coordinate rules**: overlap is true when `reference_start < query_end` and `reference_end > query_start`. Exact candidate matching is a separate check.")

    with tabs[1]:
        st.subheader("Annotated variants")
        df = records_to_df(results)
        filter_cols = st.columns(3)
        categories = filter_cols[0].multiselect("Overlap category", sorted(df["Overlap_category"].unique()), default=sorted(df["Overlap_category"].unique()))
        chromosomes = filter_cols[1].multiselect("Chromosome", sorted(df["Chromosome"].unique()), default=[])
        gene_query = filter_cols[2].text_input("Filter gene contains", "")
        filtered = df[df["Overlap_category"].isin(categories)]
        if chromosomes: filtered = filtered[filtered["Chromosome"].isin(chromosomes)]
        if gene_query: filtered = filtered[filtered["Gene"].astype(str).str.contains(gene_query,case=False,na=False)]
        st.dataframe(filtered, use_container_width=True, hide_index=True, height=520)
        category_df = df["Overlap_category"].value_counts().rename_axis("Category").reset_index(name="Records")
        st.bar_chart(category_df.set_index("Category"))

    with tabs[2]:
        st.subheader("Input and coordinate quality control")
        st.write("**BED convention:** zero-based start, end-exclusive; valid intervals satisfy `start >= 0` and `end > start`.")
        qc_summary = pd.DataFrame([
            {"Check":"Mutation records parsed","Result":len(mut_rows)},
            {"Check":"Records after deduplication","Result":len(mutations)},
            {"Check":"Unique genomic intervals","Result":len(unique_positions)},
            {"Check":"RegulomeDB reference intervals","Result":len(reg_rows)},
            {"Check":"Common-RBP reference intervals","Result":len(rbp_rows)},
            {"Check":"TSS reference intervals","Result":len(tss_rows)},
            {"Check":"Genome build","Result":genome_build},
            {"Check":"Chromosome sizes supplied","Result":"Yes" if lengths is not None else "No"},
        ])
        st.dataframe(qc_summary,use_container_width=True,hide_index=True)
        if qc_issues:
            st.warning(f"Found {len(qc_issues)} coordinate/chromosome-length issue(s).")
            st.dataframe(pd.DataFrame(qc_issues),use_container_width=True,hide_index=True)
        else:
            st.success("No coordinate-boundary issues detected for the supplied chromosome lengths." if lengths is not None else "Basic interval checks passed. Upload chromosome lengths to validate chromosome boundaries.")
        if parse_issues:
            with st.expander(f"Parser notices ({len(parse_issues)})"):
                st.dataframe(pd.DataFrame(parse_issues),use_container_width=True,hide_index=True)
        st.warning("The genome-build selector records your declaration; it cannot prove that uploaded coordinates are truly hg38 or hg19. Verify source metadata for every input file.")

    with tabs[3]:
        st.subheader("Cohort-level statistical analysis")
        st.caption("Fisher tests compare the observed cohort with uploaded controls. Permutation tests use uniform random positions within chromosomes.")
        stat_rows, perm_rows = [], []
        if st.button("Run statistical tests", type="primary"):
            if not reg_ctrl_file or not rbp_ctrl_file:
                st.error("Upload both RegulomeDB and common-RBP control files to run Fisher tests.")
            else:
                reg_ctrl = annotate(load_mutations(reg_ctrl_rows),reg_index,rbp_index,tss_index)
                rbp_ctrl = annotate(load_mutations(rbp_ctrl_rows),reg_index,rbp_index,tss_index)
                if not reg_ctrl or not rbp_ctrl:
                    st.error("One or both control files contain no valid intervals.")
                else:
                    stat_rows = [fisher_test(results,reg_ctrl,"RegulomeDB_overlap"),fisher_test(results,rbp_ctrl,"Common_RBP_overlap")]
                    if lengths:
                        perm_rows = [{"Feature":"RegulomeDB",**permutation_test(mutations,reg_index,lengths,int(n_permutations),int(seed))},
                                     {"Feature":"Common RBP",**permutation_test(mutations,rbp_index,lengths,int(n_permutations),int(seed))}]
                    st.session_state["stats"] = stat_rows
                    st.session_state["perms"] = perm_rows
        stat_rows = st.session_state.get("stats",[]); perm_rows = st.session_state.get("perms",[])
        if stat_rows:
            st.markdown("**Fisher's exact tests**")
            s_df = pd.DataFrame(stat_rows)
            st.dataframe(s_df,use_container_width=True,hide_index=True)
            if any(x["Control_overlap"] == 0 for x in stat_rows): st.warning("At least one control set has zero overlaps; infinite odds ratios can occur. Confirm controls were generated independently and were not selected based on overlap status.")
        if perm_rows:
            st.markdown("**Chromosome-based permutation tests**")
            st.dataframe(pd.DataFrame(perm_rows),use_container_width=True,hide_index=True)
            st.warning("Uniform random positions do not control for GC content, mappability, gene density, or genomic compartment. The empirical p-value resolution depends on the number of permutations.")
        if not stat_rows: st.info("Bundled control datasets are loaded automatically; click Run statistical tests.")

    with tabs[4]:
        st.subheader("Export results")
        df = records_to_df(results)
        st.download_button("Download annotated variants CSV",df.to_csv(index=False).encode("utf-8-sig"),"genomic_mutation_annotations.csv","text/csv",use_container_width=True)
        if qc_issues:
            st.download_button("Download coordinate QC issues CSV",pd.DataFrame(qc_issues).to_csv(index=False).encode("utf-8-sig"),"coordinate_qc_issues.csv","text/csv",use_container_width=True)
        if st.session_state.get("stats"):
            stats_all = [{"Test":"Fisher exact",**x} for x in st.session_state["stats"]] + [{"Test":"Permutation",**x} for x in st.session_state.get("perms",[])]
            st.download_button("Download statistical results CSV",pd.DataFrame(stats_all).to_csv(index=False).encode("utf-8-sig"),"cohort_statistics.csv","text/csv",use_container_width=True)
        st.download_button("Download unique coordinate summary CSV",pd.DataFrame(sorted(unique_positions),columns=["Chromosome","Start_0based","End_0based"]).to_csv(index=False).encode("utf-8-sig"),"unique_genomic_intervals.csv","text/csv",use_container_width=True)
        st.caption("Exports contain the current uploaded data and computed annotations. Keep a record of input file versions and genome build for reproducibility.")


if __name__ == "__main__":
    main()
