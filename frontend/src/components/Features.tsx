import { motion } from "motion/react";
import { Map, Search, GitBranch, CheckCircle, ArrowDown } from "lucide-react";

const steps = [
  {
    number: "01",
    icon: Map,
    title: "Map",
    description: "Heimdall breaks your project into thousands of small fragments — each representing a precise piece of logic (a function, variable, file slice). Then it builds a semantic graph of your codebase, like a city map where functions are buildings and data flows are roads.",
  },
  {
    number: "02",
    icon: Search,
    title: "Scout",
    description: "Specialized AI agents travel the map, opening only the relevant code fragments to gather evidence about possible vulnerabilities. They investigate data flows, authentication patterns, and security controls.",
  },
  {
    number: "03",
    icon: GitBranch,
    title: "Strategize",
    description: "A planner agent connects the dots between Scout findings to form hypotheses — tracing paths like \"This API leaks tokens here → used without auth here\" across the entire codebase graph.",
  },
  {
    number: "04",
    icon: CheckCircle,
    title: "Confirm",
    description: "A reviewer model double-checks the evidence, validates exploitability, and promotes confirmed vulnerabilities to actionable reports with exact source code locations and remediation guidance.",
  },
];

export function Features() {
  return (
    <section className="relative py-32 overflow-hidden bg-[#E8E3D5]">
      {/* Blueprint grid */}
      <div className="absolute inset-0 opacity-20">
        <div className="absolute inset-0" style={{
          backgroundImage: `
            linear-gradient(#8B7355 1px, transparent 1px),
            linear-gradient(90deg, #8B7355 1px, transparent 1px)
          `,
          backgroundSize: '100px 100px'
        }} />
      </div>

      <div className="relative z-10 max-w-5xl mx-auto px-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <div className="inline-block px-3 py-1 border border-[#8B7355] mb-6">
            <span className="text-xs uppercase tracking-wider text-[#8B7355]">Codebase Analysis</span>
          </div>
          <h2 className="text-3xl md:text-4xl mb-6" style={{ fontFamily: 'serif' }}>
            Heimdall doesn't just scan code — it <em>investigates</em> it
          </h2>
          <p className="text-xl text-[#5A5A5A] max-w-3xl mx-auto">
            Breaking projects into semantic fragments and building dynamic graphs to trace vulnerabilities back to exact source code
          </p>
        </motion.div>

        {/* Process Steps */}
        <div className="space-y-8">
          {steps.map((step, index) => (
            <div key={index}>
              <motion.div
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6, delay: index * 0.15 }}
                className="relative"
              >
                <div className="bg-white border-2 border-[#2B2B2B] rounded-lg p-8 relative overflow-hidden">
                  {/* Corner marks */}
                  <div className="absolute top-2 left-2 w-6 h-6 border-t-2 border-l-2 border-[#8B7355]" />
                  <div className="absolute top-2 right-2 w-6 h-6 border-t-2 border-r-2 border-[#8B7355]" />
                  <div className="absolute bottom-2 left-2 w-6 h-6 border-b-2 border-l-2 border-[#8B7355]" />
                  <div className="absolute bottom-2 right-2 w-6 h-6 border-b-2 border-r-2 border-[#8B7355]" />

                  <div className="flex items-start gap-6">
                    {/* Step Number */}
                    <div className="flex-shrink-0">
                      <div className="text-6xl text-[#8B7355] opacity-30" style={{ fontFamily: 'serif' }}>
                        {step.number}
                      </div>
                    </div>

                    {/* Icon */}
                    <div className="flex-shrink-0 mt-2">
                      <div className="w-16 h-16 border-2 border-[#2B2B2B] bg-[#E8E3D5] flex items-center justify-center">
                        <step.icon className="w-8 h-8 text-[#2B2B2B]" strokeWidth={1.5} />
                      </div>
                    </div>

                    {/* Content */}
                    <div className="flex-1 pt-2">
                      <h3 className="text-3xl mb-4" style={{ fontFamily: 'serif' }}>
                        {step.title}
                      </h3>
                      <p className="text-[#5A5A5A] leading-relaxed">
                        {step.description}
                      </p>
                    </div>
                  </div>
                </div>
              </motion.div>

              {/* Arrow connector between steps */}
              {index < steps.length - 1 && (
                <motion.div
                  initial={{ opacity: 0 }}
                  whileInView={{ opacity: 1 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.4, delay: index * 0.15 + 0.3 }}
                  className="flex justify-center py-4"
                >
                  <ArrowDown className="w-8 h-8 text-[#8B7355]" strokeWidth={2} />
                </motion.div>
              )}
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
