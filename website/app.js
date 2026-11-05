const observer=new IntersectionObserver(entries=>entries.forEach(entry=>{if(entry.isIntersecting){document.querySelectorAll('nav a[href^="#"]').forEach(a=>{const current=a.getAttribute('href')===`#${entry.target.id}`;a.classList.toggle('active',current);if(current)a.setAttribute('aria-current','location');else a.removeAttribute('aria-current');});}}),{rootMargin:'-20% 0px -65% 0px'});
document.querySelectorAll('section[id]').forEach(section=>observer.observe(section));

const header=document.querySelector('.site-header');
const paperTitle=document.querySelector('#paper-title');
function updateHeaderTitle(){header.classList.toggle('show-paper-title',paperTitle.getBoundingClientRect().bottom<=0)}
window.addEventListener('scroll',updateHeaderTitle,{passive:true});
window.addEventListener('resize',updateHeaderTitle);updateHeaderTitle();
