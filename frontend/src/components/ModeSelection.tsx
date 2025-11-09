import { motion } from "motion/react";
import { Search, FileCode, ArrowRight } from "lucide-react";

export function ModeSelection() {
  return (
    <section className="relative py-32 overflow-hidden bg-white">
      {/* Technical drawing grid */}
      <div className="absolute inset-0 opacity-10">
        <div className="absolute inset-0" style={{
          backgroundImage: `linear-gradient(#2B2B2B 1px, transparent 1px), linear-gradient(90deg, #2B2B2B 1px, transparent 1px)`,
          backgroundSize: '50px 50px'
        }} />
      </div>

      <div className="relative z-10 max-w-6xl mx-auto px-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <div className="inline-block px-3 py-1 border border-[#8B7355] mb-6">
            <span className="text-xs uppercase tracking-wider text-[#8B7355]">Analysis Methods</span>
          </div>
          <h2 className="text-4xl md:text-5xl mb-4" style={{ fontFamily: 'serif' }}>Choose Your Approach</h2>
          <p className="text-xl text-[#5A5A5A]">Select the architectural analysis mode that fits your project</p>
        </motion.div>

        <div className="grid md:grid-cols-2 gap-8">
          {/* Runtime Analysis */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            whileHover={{ 
              y: -8,
            }}
            className="bg-[#E8E3D5] border-2 border-[#2B2B2B] rounded-lg p-8 cursor-pointer transition-all duration-300 relative overflow-hidden"
          >
            {/* Corner marks */}
            <div className="absolute top-2 left-2 w-6 h-6 border-t-2 border-l-2 border-[#8B7355]" />
            <div className="absolute top-2 right-2 w-6 h-6 border-t-2 border-r-2 border-[#8B7355]" />
            <div className="absolute bottom-2 left-2 w-6 h-6 border-b-2 border-l-2 border-[#8B7355]" />
            <div className="absolute bottom-2 right-2 w-6 h-6 border-b-2 border-r-2 border-[#8B7355]" />

            <div className="w-20 h-20 border-2 border-[#2B2B2B] bg-white flex items-center justify-center mb-6">
              <Search className="w-10 h-10 text-[#2B2B2B]" strokeWidth={1.5} />
            </div>
            
            <h3 className="text-2xl mb-4" style={{ fontFamily: 'serif' }}>Runtime Analysis</h3>
            
            <p className="text-[#5A5A5A] mb-8">
              Observe your application in execution. Capture traces, analyze behavior patterns, and generate blueprints from live system activity.
            </p>

            <div className="flex items-center gap-2 group">
              <span className="text-[#2B2B2B] uppercase tracking-wide text-sm">Begin Trace</span>
              <ArrowRight className="w-5 h-5 text-[#2B2B2B] group-hover:translate-x-2 transition-transform" strokeWidth={2} />
            </div>
          </motion.div>

          {/* Codebase Analysis */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            whileHover={{ 
              y: -8,
            }}
            className="bg-[#E8E3D5] border-2 border-[#2B2B2B] rounded-lg p-8 cursor-pointer transition-all duration-300 relative overflow-hidden"
          >
            {/* Corner marks */}
            <div className="absolute top-2 left-2 w-6 h-6 border-t-2 border-l-2 border-[#8B7355]" />
            <div className="absolute top-2 right-2 w-6 h-6 border-t-2 border-r-2 border-[#8B7355]" />
            <div className="absolute bottom-2 left-2 w-6 h-6 border-b-2 border-l-2 border-[#8B7355]" />
            <div className="absolute bottom-2 right-2 w-6 h-6 border-b-2 border-r-2 border-[#8B7355]" />

            <div className="w-20 h-20 border-2 border-[#2B2B2B] bg-white flex items-center justify-center mb-6">
              <FileCode className="w-10 h-10 text-[#2B2B2B]" strokeWidth={1.5} />
            </div>
            
            <h3 className="text-2xl mb-4" style={{ fontFamily: 'serif' }}>Codebase Analysis</h3>
            
            <p className="text-[#5A5A5A] mb-8">
              Parse source code directly. Build dependency graphs, detect architectural patterns, and document structure without execution.
            </p>

            <div className="flex items-center gap-2 group">
              <span className="text-[#2B2B2B] uppercase tracking-wide text-sm">Parse Codebase</span>
              <ArrowRight className="w-5 h-5 text-[#2B2B2B] group-hover:translate-x-2 transition-transform" strokeWidth={2} />
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  );
}
