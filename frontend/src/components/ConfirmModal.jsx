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
  onCancel,
}) {
  if (!isOpen) return null;

  const handleClose = onClose || onCancel;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-charcoal-900/50 backdrop-blur-xs animate-fadeIn font-sans">
      <div className="bg-parchment-50 border border-warmborder rounded-2xl shadow-2xl max-w-md w-[92vw] sm:w-full p-5 sm:p-6 space-y-4 sm:space-y-5 relative">
        {/* Close Button */}
        <button
          type="button"
          onClick={handleClose}
          className="absolute top-3.5 right-3.5 flex items-center justify-center min-w-[44px] min-h-[44px] p-2 text-charcoal-500 hover:text-charcoal-900 transition-colors rounded-xl hover:bg-parchment-200 cursor-pointer"
          aria-label="Close modal"
        >
          <X className="w-5 h-5" />
        </button>

        {/* Modal Header */}
        <div className="flex items-start gap-3.5 pr-8">
          <div className="w-10 h-10 rounded-full bg-rust-100 border border-rust-600/20 flex items-center justify-center text-rust-600 shrink-0">
            <AlertTriangle className="w-5 h-5" />
          </div>
          <div className="space-y-1 pt-0.5">
            <h3 className="font-serif font-bold text-base sm:text-lg text-charcoal-900 tracking-tight">
              {title}
            </h3>
            <p className="text-xs text-charcoal-700 leading-relaxed font-sans">
              {message}
            </p>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center justify-end gap-3 pt-3 border-t border-warmborder/80">
          <button
            type="button"
            onClick={handleClose}
            className="min-h-[44px] px-4 py-2.5 rounded-xl bg-parchment-200 hover:bg-parchment-300/60 text-charcoal-900 text-xs font-semibold border border-warmborder transition-colors cursor-pointer"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={() => {
              onConfirm();
              if (handleClose) handleClose();
            }}
            className="min-h-[44px] px-4 py-2.5 rounded-xl bg-terracotta-600 hover:bg-terracotta-700 text-parchment-50 text-xs font-semibold shadow-2xs flex items-center gap-1.5 transition-colors cursor-pointer"
          >
            <Trash2 className="w-4 h-4" />
            {confirmText}
          </button>
        </div>
      </div>
    </div>
  );
}
