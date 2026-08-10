import React from 'react';
import { NavLink, Link } from 'react-router-dom';
import { BookOpen, Shield, Sliders, HelpCircle, LayoutDashboard } from 'lucide-react';

export default function Navbar() {
  const navItems = [
    { to: '/', label: 'Workspace', icon: LayoutDashboard },
    { to: '/how-it-works', label: 'How It Works', icon: BookOpen },
    { to: '/privacy', label: 'Privacy & Your Data', icon: Shield },
    { to: '/retrieval-settings', label: 'Retrieval Settings', icon: Sliders },
    { to: '/faq', label: 'FAQ', icon: HelpCircle },
  ];

  return (
    <header className="h-14 shrink-0 bg-[#DFD5C3] border-b border-[#C8BCA8] px-6 flex items-center justify-between font-sans select-none z-30 shadow-xs">
      {/* Brand Identity */}
      <Link to="/" className="flex items-center gap-3 group">
        <img
          src="/fav-icon.png"
          alt="Ask My Documents Logo"
          className="w-8 h-8 object-contain group-hover:scale-105 transition-transform"
        />
        <span className="font-serif font-bold text-charcoal-900 text-lg tracking-tight">
          Ask My Documents
        </span>
      </Link>

      {/* Navigation Links */}
      <nav className="flex items-center gap-2 text-xs font-medium text-charcoal-700">
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
    </header>
  );
}
