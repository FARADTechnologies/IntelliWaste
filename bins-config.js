// bins-config.js
const BinAssets = {
    totalImages: 68,
    imagePath: 'bins/', // Resimlerini koyduğun klasör adı

    // Her kutu için rastgele resim seçen yardımcı fonksiyon
    getRandomImage: function() {
        const rand = Math.floor(Math.random() * this.totalImages) + 1;
        return `${this.imagePath}${rand}.jpg`;
    },

    // Detay panelini güncelleyen ve AI sayfasını açan fonksiyon
    initViewAICam: function(binId, binImg) {
        if (typeof openAI === 'function') {
            openAI(binId, binImg);
        }
    }
};