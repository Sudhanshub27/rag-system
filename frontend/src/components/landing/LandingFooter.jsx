import React from 'react';
import { Link } from 'react-router-dom';

export default function LandingFooter() {
  return (
    <footer className="border-t border-warmborder bg-parchment-200/50 py-8 sm:py-10 px-4 sm:px-6 lg:px-8 font-sans select-none">
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-6">
        {/* Brand identity */}
        <div className="flex flex-col items-center md:items-start gap-1">
          <Link to="/" className="flex items-center gap-2.5 group">
            <img
              src="/fav-icon.png"
              alt="Ask My Documents Logo"
              width="24"
              height="24"
              className="w-6 h-6 object-contain group-hover:scale-105 transition-transform"
            />
            <span className="font-serif font-bold text-charcoal-900 text-base sm:text-lg tracking-tight">
              Ask My Documents
            </span>
          </Link>
          <span className="font-sans text-xs text-charcoal-600 font-medium">
            Grounded document intelligence & retrieval
          </span>
        </div>

        {/* Links */}
        <nav aria-label="Footer Navigation" className="flex items-center gap-4 sm:gap-6 text-xs font-semibold text-charcoal-800 flex-wrap justify-center">
          <Link to="/how-it-works" className="hover:text-terracotta-700 transition-colors">
            How It Works
          </Link>
          <Link to="/retrieval-settings" className="hover:text-terracotta-700 transition-colors">
            Features
          </Link>
          <Link to="/privacy" className="hover:text-terracotta-700 transition-colors">
            Privacy
          </Link>
          <Link to="/terms" className="hover:text-terracotta-700 transition-colors">
            Terms & Conditions
          </Link>
          <Link to="/license" className="hover:text-terracotta-700 transition-colors">
            License
          </Link>
          <Link to="/faq" className="hover:text-terracotta-700 transition-colors">
            FAQ
          </Link>
          <Link to="/workspace" className="hover:text-terracotta-700 transition-colors text-terracotta-700 font-bold">
            Workspace →
          </Link>
        </nav>

        {/* Copyright & Meta */}
        <div className="text-[11px] font-mono text-charcoal-500 text-center md:text-right">
          © {new Date().getFullYear()} Ask My Documents • All Rights Reserved
        </div>
      </div>
    </footer>
  );
}
