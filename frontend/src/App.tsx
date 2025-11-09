import { Hero } from "./components/Hero";
import { ValueProp } from "./components/ValueProp";
import { ModeSelection } from "./components/ModeSelection";
import { TechOverview } from "./components/TechOverview";
import { Features } from "./components/Features";
import { Footer } from "./components/Footer";

export default function App() {
  return (
    <div className="min-h-screen bg-[#E8E3D5] text-[#2B2B2B]">
      <Hero />
      <ValueProp />
      <ModeSelection />
      <TechOverview />
      <Features />
      <Footer />
    </div>
  );
}
