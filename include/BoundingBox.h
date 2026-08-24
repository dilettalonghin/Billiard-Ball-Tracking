/**
 * @file BoundingBox.h
 *
 * @brief functions for representing a bounding box with coordinates, size, and category
 *
 * @author Antonela Nichifor
 */

#ifndef BOUNDING_BOX_H
#define BOUNDING_BOX_H

struct BoundingBox {
    int x;
    int y;
    int width;
    int height;
    int category_id;

    /**
     * @brief Constructs a BoundingBox with the given parameters.
     *
     * @param x The x-coordinate of the top-left corner.
     * @param y The y-coordinate of the top-left corner.
     * @param width The width of the bounding box.
     * @param height The height of the bounding box.
     * @param category_id The category ID of the object within the bounding box.
     */
    BoundingBox(int x, int y, int width, int height, int category_id)
        : x(x), y(y), width(width), height(height), category_id(category_id) {}
};

#endif