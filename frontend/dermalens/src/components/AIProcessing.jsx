import { motion } from "framer-motion";
import { useEffect, useState } from "react";

const stages = [
  {
    number: "01",
    title: "ANALYZE",
    description: "Image quality assessment",
  },
  {
    number: "02",
    title: "ENHANCE",
    description: "Adaptive restoration",
  },
  {
    number: "03",
    title: "VALIDATE",
    description: "AI confidence comparison",
  },
];

export default function AIProcessing({ onCancel }) {
  const [activeStage, setActiveStage] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setActiveStage((prev) => {
        if (prev < stages.length - 1) {
          return prev + 1;
        }

        return prev;
      });
    }, 1200);

    return () => clearInterval(interval);
  }, []);

  return (
    <div
      className="
        relative
        w-full
        min-h-[560px]
        rounded-[32px]
        overflow-hidden
        bg-[#030712]/90
        border
        border-white/10
        shadow-[0_0_80px_rgba(34,211,238,0.05)]
      "
    >
      {/* BACKGROUND GLOW */}
      <div
        className="
          absolute
          top-1/2
          left-1/2
          -translate-x-1/2
          -translate-y-1/2
          w-[420px]
          h-[420px]
          bg-cyan-500/[0.04]
          blur-[120px]
          rounded-full
          pointer-events-none
        "
      />

      {/* HEADER */}
      <div className="relative px-8 md:px-12 pt-10 pb-8 border-b border-white/10">
        <div className="flex items-center justify-between gap-6">
          <div>
            <p className="text-cyan-400 text-xs font-semibold tracking-[0.3em] uppercase">
              Image Reliability Analysis
            </p>

            <h2 className="text-white text-2xl md:text-3xl font-black mt-3">
              Processing Image
            </h2>

            <p className="text-slate-500 text-sm mt-2">
              DermaLens is evaluating image quality and diagnostic reliability.
            </p>
          </div>

          {/* PROCESSING INDICATOR */}
          <div className="hidden sm:flex items-center gap-2 text-xs text-cyan-400 uppercase tracking-wider">
            <motion.span
              className="w-2 h-2 rounded-full bg-cyan-400"
              animate={{
                opacity: [1, 0.3, 1],
              }}
              transition={{
                duration: 1.2,
                repeat: Infinity,
              }}
            />

            Processing
          </div>
        </div>
      </div>

      {/* PIPELINE */}
      <div className="relative px-8 md:px-12 py-10">
        <div className="max-w-4xl mx-auto">
          {stages.map((stage, index) => {
            const completed = index < activeStage;
            const active = index === activeStage;

            return (
              <div
                key={stage.number}
                className="relative"
              >
                {/* CONNECTING LINE */}
                {index < stages.length - 1 && (
                  <div className="absolute left-[19px] top-[48px] w-px h-[72px] bg-white/10">
                    <motion.div
                      className="w-full bg-cyan-400"
                      initial={{ height: "0%" }}
                      animate={{
                        height: completed ? "100%" : "0%",
                      }}
                      transition={{ duration: 0.5 }}
                    />
                  </div>
                )}

                {/* STAGE */}
                <motion.div
                  animate={{
                    opacity:
                      active || completed ? 1 : 0.45,
                  }}
                  transition={{ duration: 0.4 }}
                  className="relative flex items-start gap-5 pb-10"
                >
                  {/* NUMBER / STATUS */}
                  <motion.div
                    animate={{
                      scale: active ? [1, 1.05, 1] : 1,
                      borderColor:
                        active || completed
                          ? "rgba(34,211,238,0.45)"
                          : "rgba(255,255,255,0.10)",
                    }}
                    transition={{
                      duration: 1.2,
                      repeat: active ? Infinity : 0,
                    }}
                    className="
                      relative
                      flex-shrink-0
                      w-10
                      h-10
                      rounded-full
                      border
                      bg-white/[0.03]
                      items-center
                      justify-center
                      flex
                      z-10
                    "
                  >
                    {completed ? (
                      <span className="text-cyan-300 text-sm font-bold">
                        ✓
                      </span>
                    ) : (
                      <span
                        className={`text-xs font-bold ${
                          active
                            ? "text-cyan-300"
                            : "text-slate-600"
                        }`}
                      >
                        {stage.number}
                      </span>
                    )}

                    {/* ACTIVE RING */}
                    {active && (
                      <motion.div
                        className="
                          absolute
                          inset-[-5px]
                          rounded-full
                          border
                          border-cyan-400/20
                        "
                        animate={{
                          opacity: [0.2, 0.8, 0.2],
                        }}
                        transition={{
                          duration: 1.4,
                          repeat: Infinity,
                        }}
                      />
                    )}
                  </motion.div>

                  {/* TEXT */}
                  <div className="pt-1">
                    <div className="flex items-center gap-3">
                      <p
                        className={`text-sm font-bold tracking-[0.2em] ${
                          active || completed
                            ? "text-cyan-300"
                            : "text-slate-600"
                        }`}
                      >
                        {stage.title}
                      </p>

                      {active && (
                        <motion.span
                          animate={{
                            opacity: [0.4, 1, 0.4],
                          }}
                          transition={{
                            duration: 1.2,
                            repeat: Infinity,
                          }}
                          className="text-[10px] uppercase tracking-wider text-cyan-500"
                        >
                          In progress
                        </motion.span>
                      )}

                      {completed && (
                        <span className="text-[10px] uppercase tracking-wider text-slate-600">
                          Complete
                        </span>
                      )}
                    </div>

                    <p
                      className={`text-sm mt-2 ${
                        active || completed
                          ? "text-slate-400"
                          : "text-slate-700"
                      }`}
                    >
                      {stage.description}
                    </p>
                  </div>
                </motion.div>
              </div>
            );
          })}
        </div>
      </div>

      {/* CURRENT OPERATION */}
      <div className="relative mx-8 md:mx-12">
        <div
          className="
            rounded-2xl
            border
            border-white/10
            bg-white/[0.025]
            px-6
            py-5
          "
        >
          <div className="flex items-center justify-between gap-4">
            <div>
              <p className="text-slate-600 text-[10px] uppercase tracking-[0.2em]">
                Current Operation
              </p>

              <p className="text-slate-300 text-sm mt-2">
                {stages[activeStage].description}
              </p>
            </div>

            {/* ACTIVITY INDICATOR */}
            <div className="flex items-center gap-1">
              {[0, 1, 2].map((dot) => (
                <motion.span
                  key={dot}
                  className="w-1.5 h-1.5 rounded-full bg-cyan-400"
                  animate={{
                    opacity: [0.2, 1, 0.2],
                  }}
                  transition={{
                    duration: 0.8,
                    repeat: Infinity,
                    delay: dot * 0.2,
                  }}
                />
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* CANCEL */}
      <div className="relative flex justify-center mt-7">
        <button
          type="button"
          onClick={onCancel}
          className="
            px-6
            py-2.5
            rounded-xl
            border
            border-white/10
            bg-white/[0.025]
            text-slate-500
            text-xs
            font-medium
            uppercase
            tracking-[0.15em]
            hover:text-white
            hover:border-red-400/30
            hover:bg-red-400/[0.05]
            transition-all
            duration-300
          "
        >
          Cancel Processing
        </button>
      </div>

      {/* PIPELINE FOOTER */}
      <div className="relative border-t border-white/10 px-8 md:px-12 py-5 mt-7">
        <div className="flex items-center justify-center gap-4 text-[10px] uppercase tracking-[0.2em]">
          <span
            className={
              activeStage >= 0
                ? "text-cyan-400"
                : "text-slate-700"
            }
          >
            Analyze
          </span>

          <span className="text-slate-700">→</span>

          <span
            className={
              activeStage >= 1
                ? "text-cyan-400"
                : "text-slate-700"
            }
          >
            Enhance
          </span>

          <span className="text-slate-700">→</span>

          <span
            className={
              activeStage >= 2
                ? "text-cyan-400"
                : "text-slate-700"
            }
          >
            Validate
          </span>
        </div>
      </div>
    </div>
  );
}