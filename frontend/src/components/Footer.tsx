import { Layers } from "lucide-react";

export function Footer() {
  return (
    <footer className="relative border-t-2 border-[#2B2B2B] bg-[#E8E3D5]">
      <div className="max-w-7xl mx-auto px-6 py-12">
        <div className="flex flex-col md:flex-row justify-between items-center gap-6">
          {/* Logo and tagline */}
          <div className="flex items-center gap-3">
            <Layers className="w-8 h-8 text-[#8B7355]" strokeWidth={1.5} />
            <div>
              <div className="text-xl" style={{ fontFamily: 'serif' }}>Heimdall</div>
              <div className="text-sm text-[#6B6B6B]">
                Autonomous red-teaming for modern developers
              </div>
            </div>
          </div>

          {/* Links */}
          <div className="flex gap-8 text-sm uppercase tracking-wide">
            <a href="#" className="text-[#2B2B2B] hover:text-[#8B7355] transition-colors">
              Examples
            </a>
            <a
              href="https://github.com/wasumayan/Heimdall"
              className="text-[#2B2B2B] hover:text-[#8B7355] transition-colors"
            >
              GitHub
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
}
