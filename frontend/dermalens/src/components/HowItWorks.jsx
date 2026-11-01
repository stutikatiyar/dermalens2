import { useState } from "react";

import { motion } from "framer-motion";





const stages = [

  {

    id: "upload",

    number: "01",

    group: "FRONTEND",

    title: "Image upload",

    tech: "React + Vite",

    description:

      "The user uploads a dermatology image. The interface sends the file to the backend and later displays the original and processed result.",

    input: "Image file",

    output: "Multipart upload request",

  },

  {

    id: "api",

    number: "02",

    group: "BACKEND",

    title: "FastAPI gateway",

    tech: "POST /upload",

    description:

      "The API receives the image, validates the upload, and coordinates the image-processing pipeline.",

    input: "Uploaded image",

    output: "Validated image array",

  },

  {

    id: "quality",

    number: "03",

    group: "IMAGE ANALYSIS",

    title: "Quality assessment",

    tech: "OpenCV + metrics",

    description:

      "Baseline measurements such as brightness, sharpness, contrast, and image-quality score help characterize the input.",

    input: "Original image",

    output: "Baseline quality metrics",

  },

  {

    id: "segment",

    number: "04",

    group: "AI PROCESSING",

    title: "Lesion segmentation",

    tech: "SegFormer",

    description:

      "The segmentation model proposes a lesion mask. Reliability checks determine whether lesion-aware processing can be used.",

    input: "Image",

    output: "Mask + reliability result",

  },

  {

    id: "enhance",

    number: "05",

    group: "AI PROCESSING",

    title: "Adaptive enhancement",

    tech: "OpenCV + lesion-aware path",

    description:

      "The pipeline generates candidate restorations, such as conservative contrast enhancement, and may apply lesion-aware processing when the mask is reliable.",

    input: "Image + optional mask",

    output: "Enhancement candidates",

  },

  {

    id: "validate",

    number: "06",

    group: "VALIDATION",

    title: "Quality validation",

    tech: "Metrics + safeguards",

    description:

      "Candidates are compared with the original using image-quality and preservation metrics. The pipeline selects a candidate only when its configured quality checks pass.",

    input: "Original + candidates",

    output: "Selected result or original",

  },

  {

    id: "model",

    number: "07",

    group: "MODEL SIGNAL",

    title: "Confidence comparison",

    tech: "Downstream classifier",

    description:

      "Classifier confidence can be reported before and after processing as secondary information. It is not the primary criterion for accepting image enhancement.",

    input: "Original + selected result",

    output: "Confidence values + delta",

  },

  {

    id: "result",

    number: "08",

    group: "FRONTEND",

    title: "Results dashboard",

    tech: "React UI",

    description:

      "The interface presents the original and selected result, quality metrics, enhancement status, and model signal for inspection.",

    input: "API response",

    output: "Visual comparison + metrics",

  },

];






export default function HowItWorks() {

  const [activeId, setActiveId] = useState("validate");



  return (

    <section

      id="how-it-works"

      className="relative z-10 mx-auto w-full max-w-7xl scroll-mt-24 px-5 py-20 md:px-8 md:py-28"

    >

      <div className="pointer-events-none absolute inset-x-10 top-20 -z-10 h-72 rounded-full bg-cyan-500/[0.07] blur-3xl" />



      <motion.div

        initial={{ opacity: 0, y: 18 }}

        whileInView={{ opacity: 1, y: 0 }}

        viewport={{ once: true, amount: 0.2 }}

        transition={{ duration: 0.45 }}

        className="mb-12 max-w-3xl"

      >

        <p className="text-xs font-bold uppercase tracking-[0.32em] text-cyan-300">

          DermaLens · System architecture

        </p>

        <h2 className="mt-4 text-3xl font-black tracking-tight text-white sm:text-4xl md:text-5xl">

          From raw image to

          <span className="block text-cyan-300">quality-checked result.</span>

        </h2>

        <p className="mt-5 max-w-2xl leading-7 text-slate-400">

          Explore the end-to-end image reliability pipeline. Select any stage to

          inspect its role, inputs, outputs, and place in the architecture.

        </p>

      </motion.div>



      {/* High-level architecture */}

      <div className="rounded-[30px] border border-white/10 bg-[#071225]/85 p-5 shadow-2xl shadow-black/20 backdrop-blur-xl md:p-8">

        <div className="mb-6 flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">

          <div>

            <p className="text-lg font-bold text-white">Architecture overview</p>

            <p className="mt-1 text-sm text-slate-500">

              Select a node to view the stage details below.

            </p>

          </div>

          <span className="w-fit rounded-full border border-cyan-400/20 bg-cyan-400/[0.07] px-3 py-1.5 text-xs font-semibold text-cyan-200">

            8 pipeline stages

          </span>

        </div>



        <div className="space-y-3">

          {stages.map((stage, index) => (

            <div key={stage.id}>

              <motion.button

                type="button"

                onClick={() => setActiveId(stage.id)}

                whileHover={{ y: -1 }}

                whileTap={{ scale: 0.995 }}

                aria-pressed={activeId === stage.id}

                className={`group flex w-full flex-col gap-3 rounded-2xl border p-4 text-left transition sm:flex-row sm:items-center sm:justify-between md:px-5 ${

                  activeId === stage.id

                    ? "border-cyan-300/50 bg-cyan-400/[0.09] shadow-lg shadow-cyan-950/20"

                    : "border-white/10 bg-white/[0.025] hover:border-white/20 hover:bg-white/[0.045]"

                }`}

              >

                <div className="flex items-center gap-4">

                  <span

                    className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border text-xs font-black ${

                      activeId === stage.id

                        ? "border-cyan-300/30 bg-cyan-300/10 text-cyan-200"

                        : "border-white/10 bg-slate-900 text-slate-400"

                    }`}

                  >

                    {stage.number}

                  </span>

                  <div>

                    <p className="font-semibold text-white">{stage.title}</p>

                    <p className="mt-1 text-xs text-slate-500">{stage.tech}</p>

                  </div>

                </div>

                <div className="flex items-center justify-between gap-3 sm:justify-end">

                  <span className="text-[10px] font-bold uppercase tracking-[0.18em] text-slate-500">

                    {stage.group}

                  </span>

                  <span className={`text-lg ${activeId === stage.id ? "text-cyan-200" : "text-slate-600"}`}>

                    {activeId === stage.id ? "−" : "+"}

                  </span>

                </div>

              </motion.button>



              {activeId === stage.id && (

                <motion.div

                  initial={{ opacity: 0, height: 0 }}

                  animate={{ opacity: 1, height: "auto" }}

                  className="mx-3 mt-2 grid grid-cols-1 gap-3 rounded-xl border border-white/[0.07] bg-black/20 p-4 sm:grid-cols-2"

                >

                  <div className="sm:col-span-2">

                    <p className="text-sm leading-6 text-slate-300">{stage.description}</p>

                  </div>

                  <div className="rounded-xl border border-white/[0.07] bg-white/[0.025] p-3">

                    <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-slate-500">Input</p>

                    <p className="mt-2 text-sm text-slate-200">{stage.input}</p>

                  </div>

                  <div className="rounded-xl border border-white/[0.07] bg-white/[0.025] p-3">

                    <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-slate-500">Output</p>

                    <p className="mt-2 text-sm text-slate-200">{stage.output}</p>

                  </div>

                </motion.div>

              )}



              {index < stages.length - 1 && (

                <div className="flex h-7 items-center pl-8" aria-hidden="true">

                  <div className="h-full w-px bg-gradient-to-b from-cyan-300/40 to-blue-400/10" />

                  <span className="-ml-[5px] mt-5 text-[10px] text-cyan-300/70">▼</span>

                </div>

              )}

            </div>

          ))}

        </div>



        <div className="mt-6 grid grid-cols-1 gap-3 border-t border-white/10 pt-6 sm:grid-cols-3">

          <div className="rounded-2xl border border-white/10 bg-black/10 p-4">

            <p className="text-xs font-semibold text-slate-400">Preservation first</p>

            <p className="mt-2 text-sm leading-6 text-slate-300">

              The original remains the reference for candidate comparison.

            </p>

          </div>

          <div className="rounded-2xl border border-white/10 bg-black/10 p-4">

            <p className="text-xs font-semibold text-slate-400">Quality-led selection</p>

            <p className="mt-2 text-sm leading-6 text-slate-300">

              Configured image-quality checks drive enhancement acceptance.

            </p>

          </div>

          <div className="rounded-2xl border border-white/10 bg-black/10 p-4">

            <p className="text-xs font-semibold text-slate-400">Confidence is secondary</p>

            <p className="mt-2 text-sm leading-6 text-slate-300">

              Model confidence is reported as a signal, not as a diagnosis or the acceptance rule.

            </p>

          </div>

        </div>

      </div>



      {/* Layered system architecture */}

      <div className="mt-8 rounded-[30px] border border-white/10 bg-[#071225]/75 p-5 md:p-8">

        <div className="mb-7">

          <p className="text-lg font-bold text-white">Layered system architecture</p>

          <p className="mt-1 text-sm text-slate-500">

            Responsibilities grouped by application layer.

          </p>

        </div>



        <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">

          <ArchitectureLayer

            label="01 / PRESENTATION"

            title="Frontend"

            tech="React · Vite · Tailwind CSS"

            items={["Image upload", "Processing state", "Original/result comparison", "Metrics + status cards"]}

          />

          <ArchitectureLayer

            label="02 / ORCHESTRATION"

            title="API layer"

            tech="FastAPI · Python"

            items={["POST /upload", "File validation", "Pipeline orchestration", "Structured response"]}

          />

          <ArchitectureLayer

            label="03 / IMAGE INTELLIGENCE"

            title="Processing pipeline"

            tech="OpenCV · SegFormer · classifier"

            items={["Quality metrics", "Lesion mask + validation", "Candidate enhancement", "Candidate scoring + selection"]}

          />

        </div>



        <div className="my-5 flex flex-wrap items-center justify-center gap-2 text-xs font-semibold text-slate-400">

          <span className="rounded-full border border-white/10 px-3 py-2">Upload request</span>

          <span className="text-cyan-300">→</span>

          <span className="rounded-full border border-white/10 px-3 py-2">API + pipeline</span>

          <span className="text-cyan-300">→</span>

          <span className="rounded-full border border-white/10 px-3 py-2">Selected image + metrics</span>

          <span className="text-cyan-300">→</span>

          <span className="rounded-full border border-white/10 px-3 py-2">Results UI</span>

        </div>



        <div className="rounded-2xl border border-amber-300/15 bg-amber-300/[0.04] p-4">

          <p className="text-sm font-semibold text-amber-200">Clinical-safety boundary</p>

          <p className="mt-1 text-sm leading-6 text-slate-400">

            DermaLens assesses and improves image presentation. Enhancement metrics and classifier confidence do not establish clinical accuracy or provide a diagnosis; preserve the original image for review.

          </p>

        </div>

      </div>

    </section>

  );

}



function ArchitectureLayer({ label, title, tech, items }) {

  return (

    <div className="rounded-2xl border border-white/10 bg-white/[0.025] p-5">

      <p className="text-[10px] font-bold tracking-[0.2em] text-cyan-300">{label}</p>

      <h3 className="mt-3 text-xl font-bold text-white">{title}</h3>

      <p className="mt-1 text-xs text-slate-500">{tech}</p>

      <div className="mt-5 space-y-2">

        {items.map((item) => (

          <div key={item} className="flex items-start gap-2 text-sm text-slate-300">

            <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-cyan-300" />

            <span>{item}</span>

          </div>

        ))}

      </div>

    </div>

  );

}
