import { motion } from "motion/react";
import { ArrowRight, ArrowDown } from "lucide-react";
import surfaceAgentImg from "@/assets/surface.png";
import networkAgentImg from "@/assets/network.png";
import authEndpointAgentImg from "@/assets/endpoint.png";
import injectionAgentImg from "@/assets/injection.png";

const agents = [
  {
    image: surfaceAgentImg,
    title: "Surface Agent",
    description: "Browser automation that inspects rendered pages and DOM structure to find client-side security issues.",
  },
  {
    image: networkAgentImg,
    title: "Network Agent",
    description: "HTTP probe that checks headers, redirects, and CORS configuration for transport-layer vulnerabilities.",
  },
  {
    image: authEndpointAgentImg,
    title: "Endpoint Agent",
    description: "Maps API endpoints from network logs to detect missing authentication and exposed sensitive data.",
  },
  {
    image: injectionAgentImg,
    title: "Injection Agent",
    description: "Fuzzes input fields and parameters to find XSS vulnerabilities with context-aware payloads.",
  },
];

export function TechOverview() {
  return (
    <section className="relative py-32 overflow-hidden bg-white">
      {/* Blueprint grid */}
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
            <span className="text-xs uppercase tracking-wider text-[#8B7355]">Runtime Analysis</span>
          </div>
          <h2 className="text-4xl md:text-5xl mb-6" style={{ fontFamily: 'serif' }}>
            Four Specialized Agents
          </h2>
          <p className="text-xl text-[#5A5A5A] max-w-3xl mx-auto">
            Intelligent browser agents that simulate real user interactions to uncover vulnerabilities during runtime
          </p>
        </motion.div>

        {/* Agents Grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
          {agents.map((agent, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6, delay: index * 0.1 }}
              className="relative group"
            >
              <div className="text-center">
                {/* Agent image */}
                <div className="relative mb-6 h-96 flex items-center justify-center">
                  <motion.div
                    whileHover={{ 
                      scale: 1.05,
                    }}
                    transition={{ duration: 0.3 }}
                    className="relative w-96 h-96 flex items-center justify-center"
                  >
                    <img src={agent.image} alt={agent.title} className="w-full h-full object-contain opacity-90 mix-blend-multiply" />
                  </motion.div>
                </div>

                {/* Title */}
                <h3 className="text-2xl mb-4 group-hover:text-[#8B7355] transition-colors" style={{ fontFamily: 'serif' }}>
                  {agent.title}
                </h3>

                {/* Description */}
                <p className="text-[#5A5A5A] min-h-[120px]">
                  {agent.description}
                </p>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
