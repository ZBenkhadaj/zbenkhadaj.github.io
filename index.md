---
layout: default
title: "Ziyad BENKHADAJ"
---

<!-- Language Selector (Flags for English and French) -->
<div id="language-toggle">
  <button onclick="setLanguage('en')">
    <img src="https://upload.wikimedia.org/wikipedia/commons/8/83/Flag_of_the_United_Kingdom_%283-5%29.svg" alt="English" style="width: 30px; height: 20px;">
    English
  </button>
  <button onclick="setLanguage('fr')">
    <img src="https://upload.wikimedia.org/wikipedia/commons/c/c3/Flag_of_France.svg" alt="Français" style="width: 30px; height: 20px;">
    Français
  </button>
</div>

<!-- English Content (Default hidden) -->
<div id="content-en" class="language-content">
  <h1>Welcome to my website</h1>
  <p>This is my personal website where you can find various information about me and the services I offer.</p>
  
  <ul>
    <li>CV</li>
    <li>Philosophical opinions</li>
    <li>Math and physics tutoring, from high school to higher education (preparatory classes, university, medical school...)</li>
  </ul>

  <h2>For contact, please email me at:</h2>
  <p>Email: <a href="mailto:ziyad.benkhadaj@gmail.com">ziyad.benkhadaj@gmail.com</a></p>
</div>

<!-- French Content (Default hidden) -->
<div id="content-fr" class="language-content" style="display:none;">
  <h1>Bienvenue sur mon site web</h1>
  <p>Ceci est mon site personnel où vous pouvez trouver diverses informations sur moi et les services que je propose.</p>
  
  <ul>
    <li>CV</li>
    <li>Opinions philosophiques</li>
    <li>Soutien scolaire en mathématiques et physique, du lycée jusqu'au supérieur (prépa, licence, médecine...)</li>
  </ul>

  <h2>Pour me contacter, merci de m'écrire à l'adresse suivante :</h2>
  <p>Email : <a href="mailto:ziyad.benkhadaj@gmail.com">ziyad.benkhadaj@gmail.com</a></p>
</div>

<!-- JavaScript to Toggle Language -->
<script>
  function setLanguage(lang) {
    // Hide both language contents
    document.getElementById('content-en').style.display = 'none';
    document.getElementById('content-fr').style.display = 'none';

    // Show the selected language content
    if (lang === 'en') {
      document.getElementById('content-en').style.display = 'block';
    } else if (lang === 'fr') {
      document.getElementById('content-fr').style.display = 'block';
    }
  }

  // Set default language to English
  setLanguage('en');
</script>
