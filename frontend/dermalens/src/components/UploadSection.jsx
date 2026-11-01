import { motion } from "framer-motion";

export default function UploadSection({ handleChange }) {
  return (
    <motion.section
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
      className="relative z-10 flex justify-center px-6 -mt-10 pb-24"
    >
      <div
        className="
          w-full
          max-w-5xl
          rounded-[32px]
          border
          border-white/10
          bg-[#020617]/75
          backdrop-blur-2xl
          shadow-[0_0_80px_rgba(34,211,238,0.06)]
          overflow-hidden
        "
      >
        {/* HEADER */}
        <div className="px-8 md:px-12 pt-10 pb-8 border-b border-white/10">
          <div className="flex flex-col md:flex-row md:items-end md:justify-between gap-6">
            <div>
              <p className="text-cyan-400 text-xs font-semibold tracking-[0.3em] uppercase">
                Image Reliability Analysis
              </p>

              <h2 className="text-white text-3xl md:text-4xl font-black mt-3 tracking-tight">
                Is this image ready for AI analysis?
              </h2>

              <p className="text-slate-400 mt-3 max-w-2xl leading-relaxed">
                DermaLens evaluates image quality, applies adaptive
                restoration when required, and validates the result against
                AI confidence.
              </p>
            </div>

            {/* STATUS */}
            <div className="flex items-center gap-2 text-xs text-slate-500 uppercase tracking-wider whitespace-nowrap">
              <span className="w-2 h-2 rounded-full bg-cyan-400 shadow-[0_0_10px_rgba(34,211,238,0.7)]" />
              Awaiting Image
            </div>
          </div>
        </div>

        {/* PIPELINE */}
        <div className="px-8 md:px-12 py-7 border-b border-white/10">
          <div className="flex items-center justify-center max-w-3xl mx-auto">
            {/* ANALYZE */}
            <div className="flex items-center gap-3">
              <div
                className="
                  w-9
                  h-9
                  rounded-full
                  border
                  border-cyan-400/40
                  bg-cyan-400/10
                  flex
                  items-center
                  justify-center
                  text-cyan-300
                  text-xs
                  font-bold
                "
              >
                01
              </div>

              <div>
                <p className="text-cyan-300 text-xs font-bold tracking-[0.18em]">
                  ANALYZE
                </p>
                <p className="hidden sm:block text-slate-500 text-[11px] mt-1">
                  Image quality
                </p>
              </div>
            </div>

            {/* LINE */}
            <div className="h-px flex-1 max-w-24 md:max-w-40 bg-gradient-to-r from-cyan-400/40 to-blue-400/20 mx-4" />

            {/* ENHANCE */}
            <div className="flex items-center gap-3">
              <div
                className="
                  w-9
                  h-9
                  rounded-full
                  border
                  border-blue-400/30
                  bg-blue-400/10
                  flex
                  items-center
                  justify-center
                  text-blue-300
                  text-xs
                  font-bold
                "
              >
                02
              </div>

              <div>
                <p className="text-blue-300 text-xs font-bold tracking-[0.18em]">
                  ENHANCE
                </p>
                <p className="hidden sm:block text-slate-500 text-[11px] mt-1">
                  Adaptive restoration
                </p>
              </div>
            </div>

            {/* LINE */}
            <div className="h-px flex-1 max-w-24 md:max-w-40 bg-gradient-to-r from-blue-400/20 to-cyan-400/30 mx-4" />

            {/* VALIDATE */}
            <div className="flex items-center gap-3">
              <div
                className="
                  w-9
                  h-9
                  rounded-full
                  border
                  border-cyan-400/30
                  bg-cyan-400/5
                  flex
                  items-center
                  justify-center
                  text-cyan-200
                  text-xs
                  font-bold
                "
              >
                03
              </div>

              <div>
                <p className="text-cyan-200 text-xs font-bold tracking-[0.18em]">
                  VALIDATE
                </p>
                <p className="hidden sm:block text-slate-500 text-[11px] mt-1">
                  AI confidence
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* UPLOAD AREA */}
        <div className="p-8 md:p-12">
          <label
            htmlFor="upload"
            className="
              group
              relative
              block
              rounded-[24px]
              border
              border-white/10
              bg-white/[0.025]
              hover:bg-white/[0.045]
              hover:border-cyan-400/25
              transition-all
              duration-300
              cursor-pointer
              overflow-hidden
            "
          >
            {/* SUBTLE BACKGROUND */}
            <div className="absolute inset-0 pointer-events-none">
              <div className="absolute top-0 left-1/4 w-64 h-64 bg-cyan-400/[0.035] blur-3xl rounded-full" />
              <div className="absolute bottom-0 right-1/4 w-64 h-64 bg-blue-500/[0.035] blur-3xl rounded-full" />
            </div>

            <div className="relative flex flex-col items-center justify-center py-16 md:py-20 px-6 text-center">
              {/* IMAGE PLACEHOLDER */}
              <div
                className="
                  w-20
                  h-20
                  rounded-2xl
                  border
                  border-cyan-400/20
                  bg-cyan-400/[0.06]
                  flex
                  items-center
                  justify-center
                  transition-all
                  duration-300
                  group-hover:border-cyan-400/40
                  group-hover:bg-cyan-400/[0.09]
                  group-hover:scale-105
                "
              >
                <svg
                  width="32"
                  height="32"
                  viewBox="0 0 24 24"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                  className="text-cyan-300"
                >
                  <path
                    d="M4 19V5C4 4.44772 4.44772 4 5 4H19C19.5523 4 20 4.44772 20 5V19C20 19.5523 19.5523 20 19 20H5C4.44772 20 4 19.5523 4 19Z"
                    stroke="currentColor"
                    strokeWidth="1.5"
                  />

                  <circle
                    cx="8.5"
                    cy="8.5"
                    r="1.5"
                    stroke="currentColor"
                    strokeWidth="1.5"
                  />

                  <path
                    d="M4 16L8 12L11 15L14 11L20 17"
                    stroke="currentColor"
                    strokeWidth="1.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              </div>

              {/* MAIN TEXT */}
              <p className="text-slate-500 text-xs uppercase tracking-[0.25em] mt-8">
                Input Image
              </p>

              <h3 className="text-white text-2xl md:text-3xl font-bold mt-3">
                Choose an image to analyze
              </h3>

              <p className="text-slate-500 mt-3 text-sm max-w-md leading-relaxed">
                Upload a dermatology image to begin the reliability
                assessment and adaptive processing pipeline.
              </p>

              {/* BUTTON */}
              <div
                className="
                  mt-8
                  px-7
                  py-3.5
                  rounded-xl
                  bg-gradient-to-r
                  from-cyan-500
                  to-blue-500
                  text-white
                  font-semibold
                  text-sm
                  shadow-[0_0_30px_rgba(34,211,238,0.15)]
                  transition-all
                  duration-300
                  group-hover:scale-105
                  group-hover:shadow-[0_0_40px_rgba(34,211,238,0.25)]
                "
              >
                Choose Image
              </div>

              {/* SUPPORTED CONDITIONS */}
              <div className="flex flex-wrap items-center justify-center gap-2 mt-8">
                <span className="px-3 py-1.5 rounded-lg border border-white/10 bg-white/[0.025] text-slate-500 text-[11px] uppercase tracking-wider">
                  Blur
                </span>

                <span className="px-3 py-1.5 rounded-lg border border-white/10 bg-white/[0.025] text-slate-500 text-[11px] uppercase tracking-wider">
                  Low Light
                </span>

                <span className="px-3 py-1.5 rounded-lg border border-white/10 bg-white/[0.025] text-slate-500 text-[11px] uppercase tracking-wider">
                  Noise
                </span>

                <span className="px-3 py-1.5 rounded-lg border border-white/10 bg-white/[0.025] text-slate-500 text-[11px] uppercase tracking-wider">
                  Compression
                </span>
              </div>
            </div>
          </label>

          {/* HIDDEN FILE INPUT */}
          <input
            id="upload"
            type="file"
            accept="image/*"
            onChange={handleChange}
            className="hidden"
          />

          {/* PIPELINE FOOTNOTE */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-3 sm:gap-6 mt-6 text-[11px] uppercase tracking-[0.15em] text-slate-600">
            <span>Quality Assessment</span>

            <span className="hidden sm:block text-cyan-500/30">→</span>

            <span>Adaptive Enhancement</span>

            <span className="hidden sm:block text-cyan-500/30">→</span>

            <span>Confidence Validation</span>
          </div>
        </div>
      </div>
    </motion.section>
  );
}