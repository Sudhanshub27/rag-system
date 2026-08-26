import React from 'react';
import { Link } from 'react-router-dom';

export default function LandingFooter() {
  return (
    <footer className="border-t border-warmborder bg-parchment-200/50 py-10 sm:py-12 px-4 sm:px-6 lg:px-8 font-sans">
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-6">
        {/* Brand identity */}
        <div className="flex flex-col items-center md:items-start gap-1">
          <Link to="/" className="flex items-center gap-2.5 group">
            <img
              src="/fav-icon.png"
              alt="Ask My Documents Logo"
              className="w-6 h-6 object-contain group-hover:scale-105 transition-transform"
            />
            <span className="font-serif font-bold text-charcoal-900 text-lg tracking-tight">
              Ask My Documents
            </span>
          </Link>
          <span className="font-sans text-xs text-charcoal-600 font-medium">
            Grounded document Q&A
          </span>
        </div>

        {/* Links */}
        <nav aria-label="Footer Navigation" className="flex items-center gap-6 text-xs sm:text-sm font-semibold text-charcoal-800 flex-wrap justify-center">
          <Link to="/how-it-works" className="hover:text-terracotta-700 transition-colors">
            How It Works
          </Link>
          <Link to="/privacy" className="hover:text-terracotta-700 transition-colors">
            Privacy
          </Link>
          <Link to="/faq" className="hover:text-terracotta-700 transition-colors">
            FAQ
          </Link>
          <Link to="/workspace" className="hover:text-terracotta-700 transition-colors">
            Workspace
          </Link>
        </nav>

        {/* Copyright & Meta */}
        <div className="text-xs font-mono text-charcoal-500 text-center md:text-right">
          © {new Date().getFullYear()} Ask My Documents • Private & Verifiable RAG
        </div>
      </div>
    </footer>
  );
}
