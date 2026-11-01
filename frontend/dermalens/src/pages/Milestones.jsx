import { motion } from "framer-motion";
import Navbar from "../components/Navbar";

const milestones = [
  { number: "01", title: "Problem Identification", text: "Defined the reliability problem caused by degraded smartphone dermatology images before diagnostic AI processing.", status: "Completed" },
  { number: "02", title: "Image Quality Analysis", text: "Built quality analysis around image characteristics such as blur, exposure, resolution and other degradation signals.", status: "Completed" },
  { number: "03", title: "Degradation Simulation", text: "Created controlled degraded-image cases to evaluate the restoration pipeline under repeatable conditions.", status: "Completed" },
  { number: "04", title: "Adaptive Enhancement", text: "Implemented condition-aware enhancement modes instead of applying one restoration operation to every image.", status: "Completed" },
  { number: "05", title: "AI Confidence Validation", text: "Compared downstream AI confidence before and after enhancement as part of the validation process.", status: "Completed" },
  { number: "06", title: "Validation Gate", text: "Added a safety mechanism that can reject an enhancement and preserve the original image when validation does not support the change.", status: "Completed" },
  { number: "07", title: "Benchmark Evaluation", text: "Ran the recorded benchmark across multiple ISIC images and six degradation conditions per image.", status: "Completed" },
  { number: "08", title: "Research Dashboard", text: "Integrating benchmark evidence, validation outcomes and development progress into the frontend.", status: "In Progress" },
  { number: "09", title: "Larger Validation Study", text: "Future work: expand the benchmark, test additional dermatology models and evaluate more real-world smartphone captures.", status: "Future" },
  { number: "10", title: "External Clinical Validation", text: "Future work: evaluate the middleware with appropriate clinical datasets and expert review before making clinical claims.", status: "Future" },
];

const statusClass = {
  Completed: "bg-green-500/10 text-green-300 border-green-500/20",
  "In Progress": "bg-cyan-500/10 text-cyan-300 border-cyan-500/20",
  Future: "bg-white/5 text-slate-400 border-white/10",
};

function Milestones() {
  return (
    <div className="min-h-screen bg-[#020617] text-white">
      <Navbar />
      <main className="max-w-5xl mx-auto px-6 py-16">
        <div className="max-w-3xl">
          <p className="text-cyan-300 text-xs tracking-[0.3em] uppercase">Research Development</p>
          <h1 className="text-4xl md:text-6xl font-black mt-4 tracking-tight">DermaLens Milestones</h1>
          <p className="text-slate-400 mt-5 text-lg leading-relaxed">
            A transparent view of how the project moved from the reliability problem to adaptive restoration, validation and benchmark evaluation.
          </p>
        </div>

        <div className="relative mt-16">
          <div className="absolute left-[23px] top-3 bottom-3 w-px bg-gradient-to-b from-cyan-400/40 via-white/10 to-transparent" />
          <div className="space-y-8">
            {milestones.map((item, index) => (
              <motion.article key={item.number} initial={{ opacity: 0, x: -20 }} whileInView={{ opacity: 1, x: 0 }} viewport={{ once: true, margin: "-50px" }} transition={{ duration: 0.45, delay: Math.min(index * 0.04, 0.25) }} className="relative pl-16">
                <div className="absolute left-0 top-1 w-12 h-12 rounded-2xl bg-[#071225] border border-cyan-400/20 flex items-center justify-center text-cyan-300 font-mono text-xs">{item.number}</div>
                <div className="bg-[#071225] border border-white/10 rounded-3xl p-6 md:p-7">
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                    <h2 className="text-xl md:text-2xl font-bold">{item.title}</h2>
                    <span className={`w-fit px-3 py-1 rounded-full border text-xs font-semibold ${statusClass[item.status]}`}>{item.status}</span>
                  </div>
                  <p className="text-slate-400 leading-relaxed mt-4">{item.text}</p>
                </div>
              </motion.article>
            ))}
          </div>
        </div>
      </main>
    </div>
  );
}

export default Milestones;
