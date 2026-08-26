import React, { useState, useEffect } from 'react';
import { NavLink, Link, useLocation } from 'react-router-dom';
import { BookOpen, Shield, HelpCircle, LayoutDashboard, Sliders, Menu, X, ArrowRight, Sparkles } from 'lucide-react';

export default function Navbar() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const location = useLocation();

  // Determine navigation mode based on current route
  const isWorkspaceMode =
    location.pathname === '/workspace';

  const publicNavItems = [
    { to: '/retrieval-settings', label: 'Features', icon: Sliders },
    { to: '/how-it-works', label: 'How It Works', icon: BookOpen },
    { to: '/privacy', label: 'Privacy', icon: Shield },
    { to: '/faq', label: 'FAQ', icon: HelpCircle },
  ];

  const workspaceNavItems = [
    { to: '/workspace', label: 'Workspace', icon: LayoutDashboard },
    { to: '/retrieval-settings', label: 'Retrieval Settings', icon: Sliders },
    { to: '/how-it-works', label: 'How It Works', icon: BookOpen },
    { to: '/privacy', label: 'Privacy', icon: Shield },
    { to: '/faq', label: 'FAQ', icon: HelpCircle },
  ];

  const currentNavItems = isWorkspaceMode ? workspaceNavItems : publicNavItems;

  // Close mobile drawer on route change
  useEffect(() => {
    setMobileMenuOpen(false);
  }, [location.pathname]);

  // Lock body scroll when mobile menu is open
  useEffect(() => {
    if (mobileMenuOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }
    return () => {
      document.body.style.overflow = '';
    };
  }, [mobileMenuOpen]);

  return (
    <header className="h-14 shrink-0 w-full max-w-full box-border bg-[#DFD5C3] border-b border-[#C8BCA8] px-4 md:px-6 flex items-center justify-between font-sans select-none z-30 shadow-xs relative">
      {/* Brand Identity */}
      <Link to="/" className="flex items-center gap-2.5 sm:gap-3 group">
        <img
          src="/fav-icon.png"
          alt="Ask My Documents Logo"
          width="32"
          height="32"
          className="w-8 h-8 object-contain shrink-0 group-hover:scale-105 transition-transform"
        />
        <span className="font-serif font-bold text-charcoal-900 text-base sm:text-lg tracking-tight truncate">
          Ask My Documents
        </span>
      </Link>

      {/* Desktop Navigation Links & CTA */}
      <div className="hidden md:flex items-center gap-3">
        <nav aria-label="Main Navigation" className="flex items-center gap-1.5 text-xs font-semibold text-charcoal-800">
          {currentNavItems.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink
                key={item.to}
                to={item.to}
                className={({ isActive }) =>
                  `inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg transition-colors ${
                    isActive
                      ? 'bg-parchment-50 text-terracotta-700 shadow-2xs font-bold border border-warmborder'
                      : 'hover:bg-parchment-200/60 hover:text-charcoal-900'
                  }`
                }
              >
                <Icon className="w-3.5 h-3.5 shrink-0 opacity-80" />
                <span>{item.label}</span>
              </NavLink>
            );
          })}
        </nav>

        {/* Primary Header Action */}
        <div className="pl-2 border-l border-warmborder">
          <Link
            to="/workspace"
            className="inline-flex items-center justify-center gap-1.5 px-4 py-1.5 rounded-lg bg-terracotta-600 hover:bg-terracotta-700 active:bg-terracotta-800 text-parchment-50 font-serif font-bold text-xs shadow-2xs hover:shadow-xs transition-all active:scale-[0.98]"
          >
            <span>Try it</span>
            <ArrowRight className="w-3.5 h-3.5 shrink-0" />
          </Link>
        </div>
      </div>

      {/* Mobile Menu Trigger Button */}
      <button
        type="button"
        onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
        className="md:hidden p-2 rounded-lg text-charcoal-800 hover:bg-parchment-200/70 active:bg-parchment-300 transition-colors focus:outline-none"
        aria-label={mobileMenuOpen ? 'Close Navigation Menu' : 'Open Navigation Menu'}
      >
        {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
      </button>

      {/* Mobile Navigation Drawer Overlay */}
      {mobileMenuOpen && (
        <div className="fixed inset-0 top-14 z-40 md:hidden flex flex-col bg-parchment-100/98 backdrop-blur-md animate-fadeIn">
          <nav className="flex flex-col p-4 space-y-2 border-b border-warmborder">
            {currentNavItems.map((item) => {
              const Icon = item.icon;
              return (
                <NavLink
                  key={item.to}
                  to={item.to}
                  className={({ isActive }) =>
                    `flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-semibold transition-colors ${
                      isActive
                        ? 'bg-parchment-50 text-terracotta-700 border border-warmborder shadow-2xs font-bold'
                        : 'text-charcoal-800 hover:bg-parchment-200/60'
                    }`
                  }
                >
                  <Icon className="w-4 h-4 text-terracotta-600" />
                  <span>{item.label}</span>
                </NavLink>
              );
            })}
          </nav>

          <div className="p-4 mt-auto space-y-3 bg-parchment-200/40">
            <Link
              to="/workspace"
              className="flex items-center justify-center gap-2 w-full py-3 rounded-xl bg-terracotta-600 hover:bg-terracotta-700 active:bg-terracotta-800 text-parchment-50 font-serif font-bold text-sm shadow-sm transition-all text-center"
            >
              <span>Try Ask My Documents</span>
              <ArrowRight className="w-4 h-4 shrink-0" />
            </Link>
          </div>
        </div>
      )}
    </header>
  );
}
