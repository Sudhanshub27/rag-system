import React from 'react';
import HeroSection from '../components/landing/HeroSection';
import TechnicalPipeline from '../components/landing/TechnicalPipeline';
import SupportedFormats from '../components/landing/SupportedFormats';
import ProviderControlsSection from '../components/landing/ProviderControlsSection';
import CitationShowcase from '../components/landing/CitationShowcase';
import ComparisonSection from '../components/landing/ComparisonSection';
import FinalCTA from '../components/landing/FinalCTA';
import LandingFooter from '../components/landing/LandingFooter';

export default function LandingPage() {
  return (
    <div className="w-full h-full overflow-x-hidden overflow-y-auto bg-parchment-100 text-charcoal-900 font-sans antialiased select-none">
      <main className="w-full overflow-x-hidden">
        <HeroSection />
        <TechnicalPipeline />
        <SupportedFormats />
        <ProviderControlsSection />
        <CitationShowcase />
        <ComparisonSection />
        <FinalCTA />
      </main>
      <LandingFooter />
    </div>
  );
}
