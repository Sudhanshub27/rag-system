import React from 'react';
import { AlertTriangle, Trash2, X } from 'lucide-react';

export default function ConfirmModal({
  isOpen,
  title,
  message,
  confirmText = 'Delete',
  confirmVariant = 'danger',
  onConfirm,
  onClose,
}) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-charcoal-900/40 backdrop-blur-xs animate-fadeIn font-sans">
      <div className="bg-parchment-50 border border-warmborder rounded-xl shadow-2xl max-w-md w-full p-6 space-y-5 relative">
        {/* Close Button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 p-1 text-charcoal-500 hover:text-charcoal-900 transition-colors rounded hover:bg-parchment-200"
        >
          <X className="w-4 h-4" />
        </button>

        {/* Modal Header */}
        <div className="flex items-start gap-3.5">
          <div className="w-10 h-10 rounded-full bg-rust-100 border border-rust-600/20 flex items-center justify-center text-rust-600 shrink-0">
            <AlertTriangle className="w-5 h-5" />
          </div>
          <div className="space-y-1 pt-0.5">
            <h3 className="font-serif font-bold text-lg text-charcoal-900 tracking-tight">
              {title}
            </h3>
            <p className="text-xs text-charcoal-700 leading-relaxed font-sans">
              {message}
            </p>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center justify-end gap-3 pt-2 border-t border-warmborder/80">
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-2 rounded-lg bg-parchment-200 hover:bg-parchment-300/60 text-charcoal-900 text-xs font-semibold border border-warmborder transition-colors"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={() => {
              onConfirm();
              onClose();
            }}
            className="px-4 py-2 rounded-lg bg-terracotta-600 hover:bg-terracotta-700 text-parchment-50 text-xs font-semibold shadow-2xs flex items-center gap-1.5 transition-colors"
          >
            <Trash2 className="w-3.5 h-3.5" />
            {confirmText}
          </button>
        </div>
      </div>
    </div>
  );
}
