export function revealApp() {
  document.documentElement.removeAttribute('data-latte-loading');
  document.getElementById('latte-loading')?.remove();
}
