// Navbar scroll shadow
const navbar = document.getElementById('mainNav');
if (navbar) {
    window.addEventListener('scroll', () => {
        navbar.classList.toggle('scrolled', window.scrollY > 10);
    });
}

// Smooth scroll
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    });
});

// Staggered fade-in on scroll
const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry, i) => {
        if (entry.isIntersecting) {
            // stagger by index within parent
            const siblings = Array.from(entry.target.parentElement.children);
            const idx = siblings.indexOf(entry.target);
            entry.target.style.transitionDelay = `${idx * 80}ms`;
            entry.target.classList.add('visible');
            observer.unobserve(entry.target);
        }
    });
}, { threshold: 0.15, rootMargin: '0px 0px -40px 0px' });

document.querySelectorAll('.feature-card, .stat').forEach(el => {
    el.classList.add('fade-target');
    observer.observe(el);
});

// Add fade-target styles dynamically
const style = document.createElement('style');
style.textContent = `
    .fade-target {
        opacity: 0;
        transform: translateY(14px);
        transition: opacity 0.5s ease, transform 0.5s ease;
    }
    .fade-target.visible {
        opacity: 1;
        transform: translateY(0);
    }
`;
document.head.appendChild(style);
