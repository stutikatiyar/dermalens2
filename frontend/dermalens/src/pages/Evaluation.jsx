import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import Navbar from "../components/Navbar";
import evaluationData from "../data/evaluationData";

const conditions = ["All", ...new Set(evaluationData.map((row) => row.Condition))];

function StatCard({ label, value, detail }) {
  return (
    <div className="bg-[#071225] border border-white/10 rounded-3xl p-6">
      <p className="text-xs uppercase tracking-[0.2em] text-slate-500">
        {label}
      </p>
      <p className="text-3xl font-black text-white mt-3">{value}</p>
      {detail && (
        <p className="text-sm text-slate-400 mt-2">
          {detail}
        </p>
      )}
    </div>
  );
}

function Evaluation() {
  const [condition, setCondition] = useState("All");
  const [decision, setDecision] = useState("All");

  const stats = useMemo(() => {
    const accepted = evaluationData.filter(
      (r) => r.Decision === "Accepted"
    ).length;

    const rejected = evaluationData.length - accepted;

    // Ignore invalid / NaN quality values
    const validQualityValues = evaluationData
      .map((r) => Number(r["Quality Improvement"]))
      .filter((value) => Number.isFinite(value));

    const avgQuality =
      validQualityValues.length > 0
        ? validQualityValues.reduce((sum, value) => sum + value, 0) /
          validQualityValues.length
        : 0;

    // Ignore invalid / NaN confidence values
    const validConfidenceValues = evaluationData
      .map((r) => Number(r["AI Confidence Delta"]))
      .filter((value) => Number.isFinite(value));

    const avgConfidence =
      validConfidenceValues.length > 0
        ? validConfidenceValues.reduce((sum, value) => sum + value, 0) /
          validConfidenceValues.length
        : 0;

    return {
      accepted,
      rejected,
      avgQuality,
      avgConfidence,
    };
  }, []);

  const filtered = evaluationData.filter(
    (row) =>
      (condition === "All" || row.Condition === condition) &&
      (decision === "All" || row.Decision === decision)
  );

  return (
    <div className="min-h-screen bg-[#020617] text-white">
      <Navbar />

      <main className="max-w-7xl mx-auto px-6 py-16">
        <div className="max-w-3xl">
          <p className="text-cyan-300 text-xs tracking-[0.3em] uppercase">
            Validation & Results
          </p>

          <h1 className="text-4xl md:text-6xl font-black mt-4 tracking-tight">
            Benchmark Evaluation
          </h1>

          <p className="text-slate-400 mt-5 text-lg leading-relaxed">
            DermaLens was evaluated across the recorded degradation benchmark.
            The validation gate compares the downstream AI response before
            accepting an enhancement.
          </p>
        </div>

        {/* STAT CARDS */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mt-12">
          <StatCard
            label="Evaluation Cases"
            value={evaluationData.length}
            detail="Recorded benchmark rows"
          />

          <StatCard
            label="Accepted"
            value={stats.accepted}
            detail={`${(
              (stats.accepted / evaluationData.length) *
              100
            ).toFixed(1)}% of cases`}
          />

          <StatCard
            label="Rejected"
            value={stats.rejected}
            detail="Validation-gate rejections"
          />

          <StatCard
            label="Avg. Quality Δ"
            value={`${stats.avgQuality >= 0 ? "+" : ""}${stats.avgQuality.toFixed(
              2
            )}`}
            detail="Across all recorded cases"
          />
        </div>

        {/* VALIDATION SUMMARY */}
        <div className="grid lg:grid-cols-2 gap-6 mt-6">
          <div className="bg-[#071225] border border-white/10 rounded-3xl p-7">
            <p className="text-xs uppercase tracking-[0.2em] text-slate-500">
              Average AI Confidence Δ
            </p>

            <p
              className={`text-4xl font-black mt-3 ${
                stats.avgConfidence >= 0
                  ? "text-green-400"
                  : "text-amber-400"
              }`}
            >
              {stats.avgConfidence >= 0 ? "+" : ""}
              {(stats.avgConfidence * 100).toFixed(3)} pp
            </p>

            <p className="text-slate-400 text-sm mt-2">
              Raw benchmark confidence delta, not a claim of clinical
              performance.
            </p>
          </div>

          <div className="bg-[#071225] border border-white/10 rounded-3xl p-7">
            <p className="text-xs uppercase tracking-[0.2em] text-slate-500">
              Validation Principle
            </p>

            <p className="text-xl font-semibold mt-3">
              Enhance only when downstream validation supports it.
            </p>

            <p className="text-slate-400 text-sm mt-2">
              Rejected cases fall back to the original image rather than
              forcing an enhancement.
            </p>
          </div>
        </div>

        {/* BENCHMARK TABLE */}
        <section className="mt-12 bg-[#071225] border border-white/10 rounded-[32px] overflow-hidden">
          <div className="p-7 border-b border-white/10 flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div>
              <h2 className="text-2xl font-bold">
                Benchmark Cases
              </h2>

              <p className="text-sm text-slate-500 mt-1">
                Showing {filtered.length} of {evaluationData.length} recorded
                cases
              </p>
            </div>

            <div className="flex flex-wrap gap-3">
              <select
                value={condition}
                onChange={(e) => setCondition(e.target.value)}
                className="bg-[#020617] border border-white/10 rounded-xl px-4 py-2 text-sm text-slate-300"
              >
                {conditions.map((item) => (
                  <option key={item}>{item}</option>
                ))}
              </select>

              <select
                value={decision}
                onChange={(e) => setDecision(e.target.value)}
                className="bg-[#020617] border border-white/10 rounded-xl px-4 py-2 text-sm text-slate-300"
              >
                <option>All</option>
                <option>Accepted</option>
                <option>Rejected</option>
              </select>
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-white/[0.03] text-slate-500 text-xs uppercase tracking-wider">
                <tr>
                  {[
                    "Image",
                    "Condition",
                    "Quality",
                    "Quality Δ",
                    "AI Confidence",
                    "AI Δ",
                    "Decision",
                    "Mode",
                  ].map((h) => (
                    <th
                      key={h}
                      className="text-left px-5 py-4 whitespace-nowrap"
                    >
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>

              <tbody>
                {filtered.map((row, index) => {
                  const q = Number(row["Quality Improvement"]);
                  const c = Number(row["AI Confidence Delta"]);

                  return (
                    <motion.tr
                      key={`${row.Image}-${row.Condition}`}
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{
                        delay: Math.min(index * 0.01, 0.3),
                      }}
                      className="border-t border-white/5 hover:bg-white/[0.025]"
                    >
                      <td className="px-5 py-4 text-slate-300 whitespace-nowrap">
                        {row.Image.replace(".jpg", "")}
                      </td>

                      <td className="px-5 py-4 text-slate-400 whitespace-nowrap">
                        {row.Condition}
                      </td>

                      <td className="px-5 py-4 whitespace-nowrap">
                        {Number(row["Quality Before"]).toFixed(2)} →{" "}
                        {Number(row["Quality After"]).toFixed(2)}
                      </td>

                      <td
                        className={`px-5 py-4 font-semibold ${
                          Number.isFinite(q)
                            ? q >= 0
                              ? "text-green-400"
                              : "text-amber-400"
                            : "text-slate-500"
                        }`}
                      >
                        {Number.isFinite(q)
                          ? `${q >= 0 ? "+" : ""}${q.toFixed(2)}`
                          : "—"}
                      </td>

                      <td className="px-5 py-4 whitespace-nowrap">
                        {(Number(row["AI Confidence Before"]) * 100).toFixed(
                          2
                        )}
                        % →{" "}
                        {(
                          Number(row["AI Confidence After"]) * 100
                        ).toFixed(2)}
                        %
                      </td>

                      <td
                        className={`px-5 py-4 ${
                          Number.isFinite(c)
                            ? c >= 0
                              ? "text-green-400"
                              : "text-amber-400"
                            : "text-slate-500"
                        }`}
                      >
                        {Number.isFinite(c)
                          ? `${c >= 0 ? "+" : ""}${(c * 100).toFixed(3)}%`
                          : "—"}
                      </td>

                      <td className="px-5 py-4">
                        <span
                          className={`px-2.5 py-1 rounded-full text-xs font-semibold ${
                            row.Decision === "Accepted"
                              ? "bg-green-500/10 text-green-300"
                              : "bg-amber-500/10 text-amber-300"
                          }`}
                        >
                          {row.Decision}
                        </span>
                      </td>

                      <td className="px-5 py-4 text-slate-500 min-w-64">
                        {row.Mode}
                      </td>
                    </motion.tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>
      </main>
    </div>
  );
}

export default Evaluation;