import React from 'react';
import HeroSection from '../components/landing/HeroSection';
import LandingShowcaseSection from '../components/landing/LandingShowcaseSection';
import FinalCTA from '../components/landing/FinalCTA';
import LandingFooter from '../components/landing/LandingFooter';

export default function LandingPage() {
  return (
    <div className="w-full h-full overflow-x-hidden overflow-y-auto bg-parchment-100 text-charcoal-900 font-sans antialiased select-none flex flex-col justify-between">
      <div className="w-full flex-1">
        <HeroSection />
        <LandingShowcaseSection />
        <FinalCTA />
      </div>
      <LandingFooter />
    </div>
  );
}
