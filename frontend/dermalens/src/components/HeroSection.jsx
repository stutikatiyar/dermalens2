import AIOrb from "./AIOrb";
import { motion } from "framer-motion";

export default function HeroSection() {
  return (
    <section className="relative min-h-screen flex items-center justify-between px-8 md:px-20 overflow-hidden">

      {/* LEFT SIDE */}
      <div className="max-w-3xl z-10">

        {/* CATEGORY */}
        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-cyan-400 uppercase tracking-[0.3em] text-sm mb-5"
        >
          AI Dermatology Image Reliability
        </motion.p>

        {/* PROBLEM STATEMENT */}
        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.15, duration: 0.6 }}
          className="text-slate-400 text-sm md:text-base font-medium tracking-wide mb-4"
        >
          LOW-QUALITY IMAGES CAN COMPROMISE AI ANALYSIS
        </motion.p>

        {/* MAIN HEADING */}
        <motion.h1
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="text-6xl md:text-8xl font-black leading-[0.9] tracking-tight text-white"
        >
          Restore
          <br />

          <span className="bg-gradient-to-r from-cyan-300 via-blue-400 to-white bg-clip-text text-transparent">
            Diagnostic
          </span>

          <br />

          <span className="bg-gradient-to-r from-blue-400 to-slate-200 bg-clip-text text-transparent">
            Clarity
          </span>
        </motion.h1>

        {/* SOLUTION DESCRIPTION */}
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4, duration: 0.8 }}
          className="mt-8 text-lg md:text-xl text-slate-400 leading-relaxed max-w-2xl"
        >
          DermaLens detects image-quality problems, adaptively enhances
          low-quality dermatology images, and validates the enhancement
          against AI confidence before accepting the result.
        </motion.p>

        {/* PROCESS HINT */}
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.55, duration: 0.8 }}
          className="mt-4 text-sm text-slate-500 tracking-wide"
        >
          Detect → Enhance → Validate
        </motion.p>

        {/* BUTTONS */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.7 }}
          className="mt-10 flex gap-5"
        >
          <button className="px-8 py-4 rounded-2xl bg-gradient-to-r from-cyan-500 to-blue-500 text-white font-semibold shadow-[0_0_40px_rgba(34,211,238,0.25)] hover:scale-105 transition-all duration-300">
            Upload Image
          </button>

          <button className="px-8 py-4 rounded-2xl border border-cyan-400/20 bg-white/5 backdrop-blur-xl text-cyan-100 hover:bg-cyan-400/10 transition-all duration-300">
            How It Works
          </button>
        </motion.div>

        {/* SYSTEM CAPABILITIES */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.9, duration: 0.6 }}
          className="flex gap-6 mt-14"
        >

          {/* ADAPTIVE ENHANCEMENT */}
          <div className="bg-white/5 border border-white/10 backdrop-blur-xl rounded-2xl px-6 py-4 min-w-[165px]">
            <p className="text-cyan-300 text-2xl font-black tracking-wide">
              ENHANCE
            </p>

            <p className="text-slate-400 text-sm mt-1">
              Adaptive Processing
            </p>
          </div>

          {/* AI VALIDATION */}
          <div className="bg-white/5 border border-white/10 backdrop-blur-xl rounded-2xl px-6 py-4 min-w-[165px]">
            <p className="text-blue-300 text-2xl font-black tracking-wide">
              VALIDATE
            </p>

            <p className="text-slate-400 text-sm mt-1">
              AI Confidence
            </p>
          </div>

        </motion.div>

      </div>

      {/* RIGHT SIDE */}
      <div className="hidden lg:flex items-center justify-center flex-1">
        <AIOrb />
      </div>

    </section>
  );
}