import SwiftUI

enum AppColors {
    static let darkGreen = Color(red: 0.0, green: 0.5, blue: 0.2)
    static let lightGreen = Color(red: 0.4, green: 0.7, blue: 0.4)
    static let buttonGray = Color(red: 0.95, green: 0.95, blue: 0.95)

    static func configureTabBarAppearance() {
        let appearance = UITabBarAppearance()
        appearance.configureWithDefaultBackground()

        let selectedColor = UIColor(darkGreen)
        let normalColor = UIColor.secondaryLabel

        let itemAppearance = UITabBarItemAppearance()
        itemAppearance.selected.iconColor = selectedColor
        itemAppearance.selected.titleTextAttributes = [.foregroundColor: selectedColor]
        itemAppearance.normal.iconColor = normalColor
        itemAppearance.normal.titleTextAttributes = [.foregroundColor: normalColor]

        appearance.stackedLayoutAppearance = itemAppearance
        appearance.inlineLayoutAppearance = itemAppearance
        appearance.compactInlineLayoutAppearance = itemAppearance

        UITabBar.appearance().standardAppearance = appearance
        UITabBar.appearance().scrollEdgeAppearance = appearance
    }
}
