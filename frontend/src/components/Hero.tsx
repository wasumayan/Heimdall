import { motion } from "motion/react";
import { Button } from "./ui/button";
import { Layers, Box, GitBranch, Grid3x3 } from "lucide-react";

export function Hero() {
  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
      {/* Blueprint background */}
      <div className="absolute inset-0 bg-[#E8E3D5]" />
      
      {/* Grid pattern - architectural paper */}
      <div className="absolute inset-0 opacity-30">
        <div className="absolute inset-0" style={{
          backgroundImage: `
            linear-gradient(#8B7355 1px, transparent 1px),
            linear-gradient(90deg, #8B7355 1px, transparent 1px),
            linear-gradient(#8B7355 0.5px, transparent 0.5px),
            linear-gradient(90deg, #8B7355 0.5px, transparent 0.5px)
          `,
          backgroundSize: '100px 100px, 100px 100px, 20px 20px, 20px 20px'
        }} />
      </div>

      {/* Paper texture overlay */}
      <div className="absolute inset-0 opacity-10" style={{
        backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 400 400' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)'/%3E%3C/svg%3E")`,
      }} />

      {/* Corner marks */}
      <div className="absolute top-8 left-8 w-16 h-16 border-l-2 border-t-2 border-[#2B2B2B]" />
      <div className="absolute top-8 right-8 w-16 h-16 border-r-2 border-t-2 border-[#2B2B2B]" />
      <div className="absolute bottom-8 left-8 w-16 h-16 border-l-2 border-b-2 border-[#2B2B2B]" />
      <div className="absolute bottom-8 right-8 w-16 h-16 border-r-2 border-b-2 border-[#2B2B2B]" />

      {/* Content */}
      <div className="relative z-10 max-w-7xl mx-auto px-6 py-20">
        <div className="text-center mb-20">
          {/* Logo/Brand */}
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="flex items-center justify-center gap-3 mb-12"
          >
            <Layers className="w-10 h-10 text-[#8B7355]" strokeWidth={1.5} />
            <span className="text-3xl" style={{ fontFamily: 'serif' }}>Heimdall</span>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
            className="mb-6"
          >
            <div className="inline-block px-4 py-1 border border-[#8B7355] mb-4">
              <span className="text-xs uppercase tracking-wider text-[#8B7355]">Autonomous Red-Teaming Platform</span>
            </div>
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="text-5xl md:text-7xl mb-8 max-w-5xl mx-auto"
            style={{ fontFamily: 'serif' }}
          >
            Modern devs ship fast —
            <br />
            <span className="italic" style={{ fontWeight: '300' }}>Heimdall</span> keeps it safe.
          </motion.h1>
          
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.3 }}
            className="text-xl md:text-2xl text-[#5A5A5A] mb-10 max-w-4xl mx-auto leading-relaxed"
          >
            Built for vibecoders and small teams: autonomous browser agents that probe live apps like real users, and an AI auditor that analyzes your codebase for logic-level risks and insecure patterns.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.5 }}
            className="flex flex-wrap gap-4 justify-center mb-20"
          >
            <Button className="bg-[#2B2B2B] hover:bg-[#3B3B3B] text-[#E8E3D5] px-8 py-6 text-lg border border-[#2B2B2B]">
              Start Security Scan
            </Button>
            <Button variant="outline" className="border-2 border-[#2B2B2B] text-[#2B2B2B] hover:bg-[#2B2B2B] hover:text-[#E8E3D5] px-8 py-6 text-lg bg-transparent">
              View Documentation
            </Button>
          </motion.div>
        </div>

        {/* Security Testing Visualization */}
        <div className="relative max-w-5xl mx-auto h-[500px]">
          {/* Central Security Report */}
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.8, delay: 0.6 }}
            className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 z-20"
          >
            <motion.div
              animate={{ 
                y: [0, -10, 0],
              }}
              transition={{ 
                duration: 4,
                repeat: Infinity,
                ease: "easeInOut"
              }}
              className="relative"
            >
              <div className="bg-white/80 backdrop-blur-sm border-2 border-[#2B2B2B] rounded p-8 w-96 shadow-xl">
                {/* Report header */}
                <div className="flex items-center justify-between mb-6 pb-4 border-b-2 border-dashed border-[#8B7355]">
                  <div>
                    <div className="text-xs uppercase tracking-wider text-[#8B7355] mb-1">Security Audit</div>
                    <div className="text-lg" style={{ fontFamily: 'serif' }}>Vulnerability Report</div>
                  </div>
                  <Layers className="w-8 h-8 text-[#8B7355]" strokeWidth={1} />
                </div>
                
                {/* Findings */}
                <div className="space-y-4">
                  <div className="flex items-center gap-3">
                    <div className="w-3 h-3 border-2 border-[#2B2B2B] bg-red-600" />
                    <div className="flex-1 h-px bg-[#2B2B2B] border-t-2 border-dashed border-[#8B7355]" />
                    <span className="text-xs uppercase text-[#6B6B6B]">Critical</span>
                  </div>
                  <div className="pl-6 space-y-2">
                    <div className="text-sm text-[#5A5A5A]">→ XSS in checkout.ts:142</div>
                    <div className="text-sm text-[#5A5A5A]">→ CORS misconfiguration</div>
                    <div className="text-sm text-[#5A5A5A]">→ Exposed API endpoint</div>
                  </div>
                </div>

                {/* Annotation */}
                <div className="mt-6 pt-4 border-t border-[#8B7355]/30">
                  <div className="text-xs text-[#8B7355] italic">
                    Scanning 3 agents · 47 endpoints tested
                  </div>
                </div>
              </div>
            </motion.div>
          </motion.div>

          {/* Floating annotation cards */}
          <motion.div
            initial={{ opacity: 0, x: -50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8, delay: 0.7 }}
            className="absolute left-0 top-16 z-10"
          >
            <motion.div
              animate={{ 
                y: [0, -8, 0],
              }}
              transition={{ 
                duration: 5,
                repeat: Infinity,
                ease: "easeInOut"
              }}
              className="bg-white/70 border border-[#8B7355] rounded p-4 w-56"
            >
              <div className="flex items-center gap-2 mb-2">
                <Box className="w-4 h-4 text-[#8B7355]" strokeWidth={1.5} />
                <span className="text-sm uppercase tracking-wide text-[#8B7355]">Surface Agent</span>
              </div>
              <div className="text-xs text-[#6B6B6B]">DOM inspection active</div>
            </motion.div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: 50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8, delay: 0.8 }}
            className="absolute right-0 top-16 z-10"
          >
            <motion.div
              animate={{ 
                y: [0, -12, 0],
              }}
              transition={{ 
                duration: 4.5,
                repeat: Infinity,
                ease: "easeInOut"
              }}
              className="bg-white/70 border border-[#8B7355] rounded p-4 w-56"
            >
              <div className="flex items-center gap-2 mb-2">
                <GitBranch className="w-4 h-4 text-[#8B7355]" strokeWidth={1.5} />
                <span className="text-sm uppercase tracking-wide text-[#8B7355]">Network Agent</span>
              </div>
              <div className="text-xs text-[#6B6B6B]">Header analysis complete</div>
            </motion.div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.9 }}
            className="absolute left-1/2 -translate-x-1/2 bottom-8 z-10"
          >
            <motion.div
              animate={{ 
                y: [0, -10, 0],
              }}
              transition={{ 
                duration: 3.5,
                repeat: Infinity,
                ease: "easeInOut"
              }}
              className="bg-white/70 border border-[#8B7355] rounded p-4 w-64"
            >
              <div className="flex items-center gap-2 mb-2">
                <Grid3x3 className="w-4 h-4 text-[#8B7355]" strokeWidth={1.5} />
                <span className="text-sm uppercase tracking-wide text-[#8B7355]">Injection Agent</span>
              </div>
              <div className="text-xs text-[#6B6B6B]">Fuzzing XSS payloads</div>
            </motion.div>
          </motion.div>
        </div>
      </div>
    </section>
  );
}
