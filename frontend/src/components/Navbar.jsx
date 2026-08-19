import React, { useState, useEffect } from 'react';
import { NavLink, Link, useLocation } from 'react-router-dom';
import { BookOpen, Shield, Sliders, HelpCircle, LayoutDashboard, Menu, X } from 'lucide-react';

export default function Navbar() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const location = useLocation();

  const navItems = [
    { to: '/', label: 'Workspace', icon: LayoutDashboard },
    { to: '/how-it-works', label: 'How It Works', icon: BookOpen },
    { to: '/privacy', label: 'Privacy & Your Data', icon: Shield },
    { to: '/retrieval-settings', label: 'Retrieval Settings', icon: Sliders },
    { to: '/faq', label: 'FAQ', icon: HelpCircle },
  ];

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
    <header className="h-14 shrink-0 bg-[#DFD5C3] border-b border-[#C8BCA8] px-4 md:px-6 flex items-center justify-between font-sans select-none z-30 shadow-xs relative">
      {/* Brand Identity */}
      <Link to="/" className="flex items-center gap-2.5 sm:gap-3 group">
        <img
          src="/fav-icon.png"
          alt="Ask My Documents Logo"
          className="w-7 h-7 sm:w-8 sm:h-8 object-contain group-hover:scale-105 transition-transform"
        />
        <span className="font-serif font-bold text-charcoal-900 text-base sm:text-lg tracking-tight truncate">
          Ask My Documents
        </span>
      </Link>

      {/* Desktop Navigation Links */}
      <nav className="hidden md:flex items-center gap-2 text-xs font-medium text-charcoal-700">
        {navItems.map((item) => {
          const Icon = item.icon;
          return (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === '/'}
              className={({ isActive }) =>
                `px-3.5 py-1.5 rounded-md flex items-center gap-2 transition-all ${
                  isActive
                    ? 'bg-parchment-50 text-terracotta-600 font-semibold border border-warmborder/80 shadow-2xs'
                    : 'hover:bg-parchment-200/80 hover:text-charcoal-900'
                }`
              }
            >
              <Icon className="w-3.5 h-3.5" />
              <span>{item.label}</span>
            </NavLink>
          );
        })}
      </nav>

      {/* Mobile Hamburger Toggle Button */}
      <button
        type="button"
        onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
        className="md:hidden flex items-center justify-center min-w-[44px] min-h-[44px] p-2 text-charcoal-800 hover:text-charcoal-900 hover:bg-parchment-200/60 rounded-xl transition-colors cursor-pointer"
        aria-label={mobileMenuOpen ? 'Close Navigation Menu' : 'Open Navigation Menu'}
      >
        {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
      </button>

      {/* Mobile Navigation Slide-over Drawer */}
      {mobileMenuOpen && (
        <div className="fixed inset-0 z-50 md:hidden flex flex-col justify-end">
          {/* Backdrop */}
          <div
            onClick={() => setMobileMenuOpen(false)}
            className="fixed inset-0 bg-charcoal-900/50 backdrop-blur-xs transition-opacity animate-fadeIn"
          />

          {/* Sliding Panel */}
          <aside className="fixed top-0 right-0 bottom-0 w-[80vw] max-w-xs bg-parchment-100 border-l border-warmborder shadow-2xl z-50 flex flex-col justify-between p-5 animate-slideInRight font-sans select-none overflow-y-auto">
            <div className="space-y-6">
              {/* Drawer Top Header */}
              <div className="flex items-center justify-between pb-4 border-b border-warmborder">
                <div className="flex items-center gap-2.5">
                  <img src="/fav-icon.png" alt="Logo" className="w-7 h-7 object-contain" />
                  <span className="font-serif font-bold text-charcoal-900 text-base">Navigation</span>
                </div>
                <button
                  onClick={() => setMobileMenuOpen(false)}
                  className="flex items-center justify-center min-w-[44px] min-h-[44px] p-2 text-charcoal-500 hover:text-charcoal-900 hover:bg-parchment-200 rounded-xl transition-colors"
                  aria-label="Close Drawer"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>

              {/* Navigation Links */}
              <nav className="flex flex-col gap-2">
                {navItems.map((item) => {
                  const Icon = item.icon;
                  return (
                    <NavLink
                      key={item.to}
                      to={item.to}
                      end={item.to === '/'}
                      onClick={() => setMobileMenuOpen(false)}
                      className={({ isActive }) =>
                        `min-h-[44px] px-4 py-3 rounded-xl flex items-center gap-3 text-sm font-semibold transition-all ${
                          isActive
                            ? 'bg-parchment-50 text-terracotta-600 border border-warmborder shadow-2xs'
                            : 'text-charcoal-800 hover:bg-parchment-200/80 hover:text-charcoal-900'
                        }`
                      }
                    >
                      <Icon className="w-4 h-4 text-terracotta-600 shrink-0" />
                      <span>{item.label}</span>
                    </NavLink>
                  );
                })}
              </nav>
            </div>

            <div className="pt-4 border-t border-warmborder text-[11px] text-charcoal-500 font-mono text-center">
              Ask My Documents • Grounded QA
            </div>
          </aside>
        </div>
      )}
    </header>
  );
}
