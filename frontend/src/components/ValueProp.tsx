import { motion } from "motion/react";
import { Zap, ShieldAlert, GitBranch } from "lucide-react";

export function ValueProp() {
  return (
    <section className="relative py-32 overflow-hidden bg-white">
      {/* Subtle grid */}
      <div className="absolute inset-0 opacity-10">
        <div className="absolute inset-0" style={{
          backgroundImage: `linear-gradient(#2B2B2B 1px, transparent 1px), linear-gradient(90deg, #2B2B2B 1px, transparent 1px)`,
          backgroundSize: '50px 50px'
        }} />
      </div>

      <div className="relative z-10 max-w-7xl mx-auto px-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-12"
        >
          <div className="inline-block px-3 py-1 border border-[#8B7355] mb-6">
            <span className="text-xs uppercase tracking-wider text-[#8B7355]">Core Principle</span>
          </div>
          
          <h2 className="text-4xl md:text-5xl mb-8 max-w-4xl mx-auto" style={{ fontFamily: 'serif' }}>
            Ship Fast Without Shipping Blind
          </h2>
        </motion.div>

        <div className="max-w-5xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="bg-[#E8E3D5] border-2 border-[#2B2B2B] rounded-lg p-8 md:p-12 mb-12 relative"
          >
            {/* Corner marks */}
            <div className="absolute top-2 left-2 w-6 h-6 border-t-2 border-l-2 border-[#8B7355]" />
            <div className="absolute top-2 right-2 w-6 h-6 border-t-2 border-r-2 border-[#8B7355]" />
            <div className="absolute bottom-2 left-2 w-6 h-6 border-b-2 border-l-2 border-[#8B7355]" />
            <div className="absolute bottom-2 right-2 w-6 h-6 border-b-2 border-r-2 border-[#8B7355]" />

            <div className="text-lg leading-relaxed text-[#2B2B2B] space-y-6">
              <p>
                The hypergrowth of <span className="italic">vibecoding</span> and no/low-code tools has unleashed an unprecedented wave of software creation—apps ship faster than ever, often by small teams or solo developers who deploy directly to production without traditional security review.
              </p>
              
              <p>
                Yet this acceleration has outpaced how we test and secure code.
              </p>

              <div className="border-l-4 border-[#8B7355] pl-6 my-8">
                <p className="text-xl" style={{ fontFamily: 'serif' }}>
                  <strong>Heimdall</strong> was built for this new reality: an autonomous, explainable red-teaming system that continuously simulates real user behavior, discovers runtime vulnerabilities, and maps them back to the exact lines of code responsible.
                </p>
              </div>

              <p>
                By combining browser-authentic agents with a knowledge graph of evidence and code paths, Heimdall transforms security from a reactive, manual process into an intelligent layer of development itself—so teams can move fast without shipping blind.
              </p>
            </div>

            {/* Annotation */}
            <div className="mt-8 pt-6 border-t border-[#8B7355]/30 text-xs text-[#8B7355] italic text-center">
              Continuous security testing integrated into the development workflow
            </div>
          </motion.div>

          {/* Key principles */}
          <div className="grid md:grid-cols-3 gap-6">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6, delay: 0.3 }}
              className="bg-white border border-[#2B2B2B] rounded p-6"
            >
              <div className="w-12 h-12 border border-[#8B7355] bg-[#E8E3D5]/50 flex items-center justify-center mb-4">
                <Zap className="w-6 h-6 text-[#2B2B2B]" strokeWidth={1.5} />
              </div>
              <h3 className="text-lg mb-2 uppercase tracking-wide">Autonomous</h3>
              <p className="text-sm text-[#6B6B6B]">No manual test writing — agents simulate real user behavior automatically</p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6, delay: 0.4 }}
              className="bg-white border border-[#2B2B2B] rounded p-6"
            >
              <div className="w-12 h-12 border border-[#8B7355] bg-[#E8E3D5]/50 flex items-center justify-center mb-4">
                <ShieldAlert className="w-6 h-6 text-[#2B2B2B]" strokeWidth={1.5} />
              </div>
              <h3 className="text-lg mb-2 uppercase tracking-wide">Explainable</h3>
              <p className="text-sm text-[#6B6B6B]">Evidence-linked findings with exact code locations and remediation paths</p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6, delay: 0.5 }}
              className="bg-white border border-[#2B2B2B] rounded p-6"
            >
              <div className="w-12 h-12 border border-[#8B7355] bg-[#E8E3D5]/50 flex items-center justify-center mb-4">
                <GitBranch className="w-6 h-6 text-[#2B2B2B]" strokeWidth={1.5} />
              </div>
              <h3 className="text-lg mb-2 uppercase tracking-wide">Graph-Driven</h3>
              <p className="text-sm text-[#6B6B6B]">Knowledge graphs tie runtime findings directly to repos, branches, and owners</p>
            </motion.div>
          </div>
        </div>
      </div>
    </section>
  );
}
